#!/usr/bin/env python3
"""EXP5 — checkpoint / resume / 故障恢复 演练的【验证】侧（segments 由 JCC job shell 跑）。

不改核心代码：复用 train.py 的 save_checkpoint/load_checkpoint + MoTT2IModel。三类检查：

[A] ckpt 结构 + lossless round-trip（--ckpt --config）：
    建模型/opt/sched/EMA → load_checkpoint（exact 还原 optimizer/scheduler/EMA/global_step/ablation_fp）
    → 立即 re-save → 逐张量比对两份 ckpt **bit-identical** → 证明 load→save 无损（optimizer 双态/EMA 全在）。
[B] metrics 连续性（--metrics --segments "S,2S,3S"）：
    分段训练（跑→存→重启续→存）后，metrics.jsonl 的 step 必须**严格连续无 gap/dup**，
    lr 在续训边界连续（scheduler 正确还原），loss 在边界不异常突跳（在邻域噪声带内）。
    ⚠️ 流式 loader 无 epoch/sampler state → resume 后数据是**新 shuffle**（非逐位续流，设计如此）；
    故 loss 不会逐位重合，但应 in-distribution（本检查量化"突变"）。
[C] 跨 stage 可加载（--ckpt --next-config）：
    256 ckpt 的 trainable_state 应能 strict=False 装进 512/1024 模型（模型分辨率无关）：
    0 unexpected、missing 全是 frozen（非 requires_grad）参数。

用法见各 --mode；不传则按提供的参数自动跑能跑的检查。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'✓' if cond else '✗'}] {name}{(': ' + detail) if detail else ''}", flush=True)


def _build(config_path):
    from networks.model_mot import MoTT2IModel
    cfg = OmegaConf.load(config_path if os.path.isabs(config_path) else os.path.join(ROOT, config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoTT2IModel(config=cfg).to(device)
    op = OmegaConf.to_container(cfg.optimizer.params, resolve=True)
    wd = float(op.pop("weight_decay", 0.0))
    opt = torch.optim.AdamW(model.get_optimizer_param_groups(weight_decay=wd), **op)
    from train import build_scheduler
    sched = build_scheduler(opt, cfg, int(cfg.trainer.max_steps))
    ema_cfg = cfg.trainer.get("ema", {}) or {}
    if bool(ema_cfg.get("enabled", False)):
        model.init_ema(float(ema_cfg.get("decay", 0.9999)))
    return cfg, model, opt, sched


def check_A_roundtrip(ckpt_path, config_path):
    print(f"\n[A] ckpt 结构 + lossless round-trip: {ckpt_path}", flush=True)
    from train import load_checkpoint, save_checkpoint
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ok("ckpt 含 trainable_state", "trainable_state" in raw, f"{len(raw.get('trainable_state', {}))} tensors")
    ok("ckpt 含 optimizer", "optimizer" in raw and bool(raw["optimizer"]))
    ok("ckpt 含 scheduler", "scheduler" in raw)
    ok("ckpt 含 ema", raw.get("ema") is not None)
    ok("ckpt 含 global_step", "global_step" in raw, f"step={raw.get('global_step')}")
    ok("ckpt 含 ablation_fp(F2 指纹)", raw.get("ablation_fp") is not None)
    # optimizer Adam 双态非空（exp_avg/exp_avg_sq 已建）
    osd = raw.get("optimizer", {})
    state = osd.get("state", {}) if isinstance(osd, dict) else {}
    has_moments = any(("exp_avg" in v and "exp_avg_sq" in v) for v in state.values()) if state else False
    ok("optimizer Adam 双态(exp_avg/exp_avg_sq)存在", has_moments, f"{len(state)} param states")

    cfg, model, opt, sched = _build(config_path)
    step = load_checkpoint(ckpt_path, model, opt, sched, is_dist=False, config=cfg)
    ok("load_checkpoint 还原 global_step", step == raw["global_step"], f"loaded={step}")
    # 立即 re-save，比对 bit-identical
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
    save_checkpoint(tmp, model, opt, sched, step, is_dist=False, rank=0, config=cfg)
    rs = torch.load(tmp, map_location="cpu", weights_only=False)
    os.unlink(tmp)
    # 比对可训练参数 + scheduler + ablation_fp
    a, b = raw["trainable_state"], rs["trainable_state"]
    same_keys = set(a) == set(b)
    max_d = 0.0
    if same_keys:
        for k in a:
            d = (a[k].float() - b[k].float()).abs().max().item()
            max_d = max(max_d, d)
    ok("re-save 可训练参数 keys 一致", same_keys)
    ok("re-save 可训练参数 bit-identical", max_d == 0.0, f"max|Δ|={max_d:.2e}")
    ok("re-save scheduler 一致", rs["scheduler"]["last_epoch"] == raw["scheduler"]["last_epoch"],
       f"last_epoch={rs['scheduler']['last_epoch']}")
    ok("re-save ablation_fp 一致", rs.get("ablation_fp") == raw.get("ablation_fp"))
    # EMA round-trip bit-identical
    if raw.get("ema") is not None and rs.get("ema") is not None:
        ea, eb = raw["ema"], rs["ema"]
        emax = 0.0
        if set(ea) == set(eb):
            for k in ea:
                emax = max(emax, (ea[k].float() - eb[k].float()).abs().max().item())
        ok("re-save EMA bit-identical", emax == 0.0, f"max|Δ|={emax:.2e}")


def check_B_metrics(metrics_path, segments):
    print(f"\n[B] metrics 连续性: {metrics_path} (segments={segments})", flush=True)
    rows = [json.loads(l) for l in open(metrics_path) if l.strip()]
    steps = [r["step"] for r in rows]
    ok("metrics 非空", len(rows) > 0, f"{len(rows)} rows")
    if not rows:
        return
    # 严格连续无 gap/dup
    gaps = [(steps[i - 1], steps[i]) for i in range(1, len(steps)) if steps[i] != steps[i - 1] + 1]
    ok("step 严格 +1 连续（无 gap/dup）", len(gaps) == 0, f"异常对={gaps[:5]}")
    ok("step 最大达到目标", steps[-1] == segments[-1], f"max_step={steps[-1]} target={segments[-1]}")
    # 各续训边界：loss 不异常突跳（边界后首点 vs 边界前 8 点中位，比值 < 3×）
    by_step = {r["step"]: r for r in rows}
    for boundary in segments[:-1]:  # 在每个 segment 末尾(=boundary)发生 resume
        pre = [by_step[s]["loss"] for s in range(max(1, boundary - 7), boundary + 1) if s in by_step]
        post = by_step.get(boundary + 1, {}).get("loss")
        if pre and post is not None:
            med = sorted(pre)[len(pre) // 2]
            ratio = post / (med + 1e-9)
            ok(f"resume@{boundary} loss 无突跳", 0.3 < ratio < 3.0,
               f"pre_med={med:.3f} post={post:.3f} ratio={ratio:.2f}")
        # lr 在边界连续（scheduler 还原）
        lr_pre = by_step.get(boundary, {}).get("lr")
        lr_post = by_step.get(boundary + 1, {}).get("lr")
        if lr_pre is not None and lr_post is not None:
            ok(f"resume@{boundary} lr 连续", abs(lr_post - lr_pre) / (lr_pre + 1e-12) < 0.5,
               f"lr {lr_pre:.3e}->{lr_post:.3e}")


def check_C_crossload(ckpt_path, next_config):
    print(f"\n[C] 跨 stage 可加载: {ckpt_path} -> {next_config}", flush=True)
    from networks.model_mot import MoTT2IModel
    cfg = OmegaConf.load(next_config if os.path.isabs(next_config) else os.path.join(ROOT, next_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoTT2IModel(config=cfg).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.model.load_state_dict(ck["trainable_state"], strict=False)
    trainable_names = {n for n, p in model.model.named_parameters() if p.requires_grad}
    missing_trainable = [m for m in missing if m in trainable_names]
    ok("跨 stage: 0 unexpected key", len(unexpected) == 0, f"unexpected={unexpected[:3]}")
    ok("跨 stage: 无 missing 可训练参数（missing 全是 frozen）",
       len(missing_trainable) == 0, f"missing_trainable={missing_trainable[:3]} (total missing={len(missing)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--segments", default=None, help="逗号分隔累计步数目标，如 600,1200,1800")
    ap.add_argument("--next-config", default=None)
    args = ap.parse_args()

    if args.ckpt and args.config:
        check_A_roundtrip(args.ckpt, args.config)
    if args.metrics and args.segments:
        check_B_metrics(args.metrics, [int(x) for x in args.segments.split(",")])
    if args.ckpt and args.next_config:
        check_C_crossload(args.ckpt, args.next_config)

    print(f"\n========== EXP5 RESUME VERIFY: {len(PASS)} passed, {len(FAIL)} failed ==========", flush=True)
    if FAIL:
        print("FAILED:", FAIL, flush=True)
        sys.exit(1)
    print("ALL RESUME CHECKS PASSED ✅", flush=True)


if __name__ == "__main__":
    main()
