"""取图 + 预处理层。三域三种取法（数据交付文档 §1.3 / §6.1 / §11）：

  GPIC      —— 本地文件 {src_root}/images/train/{src_shard}/{src_key}.{ext}（ext 混合 .jpg/.png）
  MONET/synth —— webdataset .tar 成员 {src_key}.jpg（全分辨率；parquet thumbnail 仅 384px 不可用）
                 .tar 在 S3 mount 上无随机索引 → **顺序流式**读（'r|'，避免 3000 header 随机 seek）
  anime     —— R2 HTTP（s3key URL，img.metanomaly.co），urllib 取 + 重试 + 可选磁盘缓存

预处理：decode→RGB；resize 短边到 target（elig 已保证 min(w,h)≥target → 只下采样，符合"不上采样"）
        →center-crop target×target→[-1,1] CHW tensor。AR 分桶留作后续增量（需分桶 batch）。
"""

from __future__ import annotations

import io
import os
import socket
import tarfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

import numpy as np
import torch
from PIL import Image

from .pil_safety import PIL_NATIVE_LOCK

Image.MAX_IMAGE_PIXELS = None                       # 关掉 decompression bomb 警告（数据可信）

_GPIC_EXTS = (".jpg", ".png", ".jpeg", ".webp")

# --------------------------------------------------------------------------- #
# anime R2 取图可观测性（review：之前失败被静默吞掉，无法区分死链/限流/超时）。
#   FETCH_STATS 累计 ok / cache_hit / fail + 失败原因(HTTP码/timeout/URLErr)；
#   失败按 _WARN_EVERY 采样打印累计成功率 + top 原因 → 在线训练可据此监控 anime 取图健康。
# --------------------------------------------------------------------------- #
FETCH_STATS: Counter = Counter()
_FETCH_LOCK = threading.Lock()
_WARN_EVERY = 200                                   # 每累计这么多次失败打一条采样日志


def _classify_fetch_error(e: Exception) -> str:
    """归类取图异常 → 'HTTP429'/'HTTP404'/'timeout'/'URLErr:ConnectionResetError' 等。"""
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP{e.code}"
    if isinstance(e, urllib.error.URLError):
        reason = getattr(e, "reason", e)
        if isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason):
            return "timeout"
        return f"URLErr:{type(reason).__name__}"
    if isinstance(e, (socket.timeout, TimeoutError)):
        return "timeout"
    # OSError 族带上 errno（2026-07-03：主训练出现 ERR:OSError 风暴但无法区分 ENOSPC 磁盘满/
    # EMFILE fd 耗尽/其它 → 加 errno 定位。e.g. "ERR:OSError.e28"=No space left on device）
    eno = getattr(e, "errno", None)
    return f"ERR:{type(e).__name__}" + (f".e{eno}" if eno else "")


def fetch_stats_snapshot() -> dict:
    """返回当前累计取图统计快照（ok/cache_hit/fail/各失败原因 + 成功率），供监控/日志。"""
    with _FETCH_LOCK:
        s = dict(FETCH_STATS)
    got = s.get("ok", 0) + s.get("cache_hit", 0)
    tried = got + s.get("fail", 0)
    s["success_rate"] = round(got / tried, 4) if tried else None
    return s


# --------------------------------------------------------------------------- #
# 取原始图字节
# --------------------------------------------------------------------------- #
def gpic_path(src_root: str, src_shard: str, src_key: str, ext: str | None = None) -> str | None:
    """GPIC 本地图路径。ext 已知（来自源 s3key，权威）→ 直接拼路径，**不再前置 os.path.exists**
    （挂载上每次 stat 是一次往返；缺文件交给 open 失败处理 → 省掉逐文件多余 stat，审计 M7）。
    ext 未知才探测 .jpg/.png/…（manifest 不带 ext）。"""
    base = os.path.join(src_root, "images", "train", str(src_shard), str(src_key))
    if ext:
        return base + (ext if ext.startswith(".") else "." + ext)
    for e in _GPIC_EXTS:
        if os.path.exists(base + e):
            return base + e
    return None


def read_gpic_bytes(src_root, src_shard, src_key, ext=None) -> bytes | None:
    p = gpic_path(src_root, src_shard, src_key, ext)
    if p is None:
        return None
    try:
        with open(p, "rb") as f:
            return f.read()
    except OSError:                                     # ext 已知时不前置 stat，open 失败（缺文件/坏挂载）→ None
        return None


def monet_tar_path(src_root: str, src_shard: str) -> str:
    """MONET/synth：src_shard 的 .parquet → 同名 .tar。"""
    return os.path.join(src_root, src_shard.replace(".parquet", ".tar"))


def stream_tar_images(tar_path: str, wanted_keys: set[str]):
    """顺序流式读一个 .tar，yield (src_key, jpg_bytes)，只产出 wanted_keys 中的成员。
    用 'r|'（streaming，无随机 seek）—— S3 mount 上比 getmember(随机索引) 快得多。
    成员名形如 {key}.jpg / {key}.json / {key}.*.pth，只取 .jpg。"""
    remaining = set(wanted_keys)
    try:
        with tarfile.open(tar_path, "r|") as tf:        # 流式，不建全量索引
            for m in tf:
                if not remaining:
                    break
                if not m.isfile() or not m.name.endswith(".jpg"):
                    continue
                key = m.name[:-4].split("/")[-1]
                if key in remaining:
                    fobj = tf.extractfile(m)
                    if fobj is not None:
                        yield key, fobj.read()
                        remaining.discard(key)
    except (FileNotFoundError, tarfile.TarError, OSError):
        return


def fetch_url_bytes(url: str, retries: int = 3, timeout: int = 20,
                    cache_dir: str | None = None) -> bytes | None:
    """anime R2 HTTP 取图。可选磁盘缓存（key=url 末段）。"""
    cache_path = None
    if cache_dir:
        # 缓存读 = best-effort（2026-07-03 硬化）：磁盘异常 / janitor 轮换目录把文件清走的竞态
        # → 落到 HTTP 正常取图，不再把整个 fetch 判失败（旧行为会连带 producer 跳过该 shard）。
        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, url.rsplit("/", 1)[-1])
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    data = f.read()
                with _FETCH_LOCK:
                    FETCH_STATS["cache_hit"] += 1
                return data
        except OSError as ce:
            with _FETCH_LOCK:
                FETCH_STATS[f"cache_read_fail:e{getattr(ce, 'errno', 0)}"] += 1
    last_reason = "unknown"
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mot-loader/1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if cache_path:
                # 缓存写 = best-effort（2026-07-03 硬化）：ENOSPC 磁盘满/巨目录退化时旧行为会把
                # 【已成功取到的图】判为取图失败丢弃（主训练 fail rate 0.2%→9% 的 ERR:OSError 风暴
                # 即此）。现在写失败只计数，不影响本次成功取图。
                try:
                    tmp = cache_path + ".tmp"
                    with open(tmp, "wb") as f:
                        f.write(data)
                    os.replace(tmp, cache_path)
                except OSError as ce:
                    with _FETCH_LOCK:
                        FETCH_STATS[f"cache_write_fail:e{getattr(ce, 'errno', 0)}"] += 1
            with _FETCH_LOCK:
                FETCH_STATS["ok"] += 1
            return data
        except Exception as e:                          # noqa: BLE001 网络抖动重试
            last_reason = _classify_fetch_error(e)
            # 4xx 客户端错误（403=CF 反爬挑战 / 404=不存在）重试无意义：CF 在短窗口内对同 URL
            # 缓存挑战裁决，立即重试是同一结果（实测同 key 60s 内才翻面）。不睡、不重试，立即放弃——
            # producer 之后 shard 重洗会重新撞到该 key（= 新一次 CF 轮盘），靠重洗恢复而非原地空睡 1.8s。
            # 仅 408/429（超时/限流）与 5xx、网络抖动才退避重试。最后一次尝试也不再尾随空睡。
            code = getattr(e, "code", None)
            nonretry = isinstance(code, int) and 400 <= code < 500 and code not in (408, 429)
            if nonretry or i == retries - 1:
                break
            time.sleep(0.3 * (i + 1))
    # 重试耗尽 → 记原因 + 采样打印累计成功率（不再静默吞掉）
    with _FETCH_LOCK:
        FETCH_STATS["fail"] += 1
        FETCH_STATS[f"reason:{last_reason}"] += 1
        nfail = FETCH_STATS["fail"]
        ok = FETCH_STATS.get("ok", 0)
        do_warn = (nfail % _WARN_EVERY == 0)
    if do_warn:
        rate = ok / (ok + nfail) if (ok + nfail) else 0.0
        top = ", ".join(f"{k[7:]}={v}" for k, v in
                        sorted(((k, v) for k, v in FETCH_STATS.items() if k.startswith("reason:")),
                               key=lambda kv: -kv[1])[:4])
        print(f"[image_fetch] ⚠️ anime 取图累计 ok={ok} fail={nfail} (成功率 {rate:.1%}); "
              f"失败原因 top: {top} | 最近: {last_reason}", flush=True)
    return None


# --------------------------------------------------------------------------- #
# decode + 预处理
# --------------------------------------------------------------------------- #
def decode_rgb(raw: bytes) -> Image.Image | None:
    try:
        with PIL_NATIVE_LOCK:
            with Image.open(io.BytesIO(raw)) as img:
                return img.convert("RGB")
    except Exception:                                   # noqa: BLE001 损坏图跳过
        return None


def crop_offset(excess: int, anchor: float = 0.5) -> int:
    """裁剪起点（竖直/水平通用）：excess=可裁像素(尺寸-target)，anchor∈[0,1]（0=贴顶/居左, 0.5=居中, 1=贴底/居右）。
    ★anchor==0.5 逐字节回落到旧 center-crop 的 excess//2（round(0.5) 的银行家舍入对奇数 excess 会破坏字节一致，故特判）。
    excess<=0 → 0（对齐旧 max(0, …)）。缺头修复(Path B, 2026-07-05)：anime portrait 顶损 → 竖直 anchor 配 <0.5 保头。"""
    if excess <= 0:
        return 0
    if anchor == 0.5:
        return excess // 2                              # 旧 center-crop 语义，逐字节不变
    return max(0, min(int(excess), int(round(excess * float(anchor)))))


def resize_crop_uint8(img: Image.Image, target: int, v_anchor: float = 0.5) -> np.ndarray:
    """resize 短边到 target（仅下采样，靠 elig 保证不上采样）→ crop target×target。
    v_anchor：竖直裁剪锚点∈[0,1]（0=贴顶保头 / 0.5=居中=旧行为 / 1=贴底）；水平方向仍居中。默认 0.5 逐字节不变。
    返回 uint8 HWC [target,target,3]（buffer 存这个省内存，batch 时再 normalize）。"""
    with PIL_NATIVE_LOCK:
        w, h = img.size
        if min(w, h) > target:                          # 只下采样
            scale = target / min(w, h)
            img = img.resize((max(target, round(w * scale)), max(target, round(h * scale))), Image.LANCZOS)
            w, h = img.size
        left = max(0, (w - target) // 2)
        top = crop_offset(h - target, v_anchor)         # 竖直锚点（默认 0.5=旧 center-crop，字节不变）
        img = img.crop((left, top, left + target, top + target))
        if img.size != (target, target):                # 兜底（短边<target 等极端）
            img = img.resize((target, target), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)          # HWC uint8


def uint8_to_norm_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 HWC → [-1,1] float CHW tensor。"""
    t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
    return t.permute(2, 0, 1).contiguous()


def process_square(img: Image.Image, target: int) -> torch.Tensor:
    """resize+center-crop+归一化 → [-1,1] float CHW tensor [3,target,target]（直用/单测）。"""
    return uint8_to_norm_tensor(resize_crop_uint8(img, target))


# --------------------------------------------------------------------------- #
# 简单 open-tar LRU（随机单成员取用；流式优先用 stream_tar_images）
# --------------------------------------------------------------------------- #
class TarMemberReader:
    """随机单成员读取（带 open-TarFile LRU）。注意：首次 getmember 会扫全量 header（S3 上慢），
    主路径请用 stream_tar_images 顺序读。"""

    def __init__(self, max_open: int = 4):
        self.max_open = max_open
        self._open: dict[str, tarfile.TarFile] = {}
        self._order: list[str] = []

    def _get(self, tar_path: str) -> tarfile.TarFile | None:
        if tar_path in self._open:
            return self._open[tar_path]
        try:
            tf = tarfile.open(tar_path, "r")            # 随机模式（建索引）
        except (FileNotFoundError, tarfile.TarError, OSError):
            return None
        self._open[tar_path] = tf
        self._order.append(tar_path)
        while len(self._order) > self.max_open:
            old = self._order.pop(0)
            try:
                self._open.pop(old).close()
            except Exception:
                pass
        return tf

    def read(self, tar_path: str, src_key: str) -> bytes | None:
        tf = self._get(tar_path)
        if tf is None:
            return None
        try:
            m = tf.getmember(f"{src_key}.jpg")
            fobj = tf.extractfile(m)
            return fobj.read() if fobj else None
        except (KeyError, tarfile.TarError, OSError):
            return None
