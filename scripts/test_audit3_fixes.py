"""验证第三份审计修复（CPU）：
  F2  ckpt 消融指纹：bidirectional↔causal（mask-only，不改参数名）等切换 resume 时报错；
      同指纹通过；旧 ckpt 无指纹只 warn 不崩。
  F5  dataset 无 tokenizer_path → 构造时 warn（退 proxy 截断）。
运行：venv-gpu/bin/python scripts/test_audit3_fixes.py
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
from train import _ablation_fingerprint, _ablation_fingerprint_check


def _cfg(**model_over):
    base = {"model": {"text_adapter": {"enabled": True, "bidirectional": True, "num_layers": 2},
                      "long_skip": {"enabled": False, "mode": "A_full"},
                      "abs_sincos": {"enabled": False}, "noise_condition": True,
                      "sandwich_norm": True, "patch_size": 1}}
    cfg = OmegaConf.create(base)
    for k, v in model_over.items():
        OmegaConf.update(cfg, f"model.{k}", v, force_add=True)
    return cfg


def test_fingerprint_bidirectional():
    fp_bidir = _ablation_fingerprint(_cfg(**{"text_adapter.bidirectional": True}))
    # 同指纹 → 通过
    _ablation_fingerprint_check(fp_bidir, _cfg(**{"text_adapter.bidirectional": True}))
    print("  ✓ 同指纹（bidirectional=True）resume 通过")
    # bidirectional 切 causal（只改 mask、不改参数名）→ 必须报错
    try:
        _ablation_fingerprint_check(fp_bidir, _cfg(**{"text_adapter.bidirectional": False}))
        raise AssertionError("bidirectional→causal 应报错")
    except RuntimeError as e:
        assert "bidirectional" in str(e), str(e)
    print("  ✓ bidirectional→causal（mask-only 切换）resume 报错（审计 F2 拦住）")


def test_fingerprint_other_switches():
    fp = _ablation_fingerprint(_cfg())
    for over in [{"noise_condition": False}, {"long_skip.enabled": True},
                 {"long_skip.mode": "B_outer"}, {"text_adapter.num_layers": 0},
                 {"abs_sincos.enabled": True}]:
        try:
            _ablation_fingerprint_check(fp, _cfg(**over))
            raise AssertionError(f"{over} 应报错")
        except RuntimeError:
            pass
    print("  ✓ noise_condition / long_skip.enabled / long_skip.mode / num_layers / abs_sincos 切换均报错")


def test_fingerprint_legacy_none():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _ablation_fingerprint_check(None, _cfg())   # 旧 ckpt 无指纹
    out = buf.getvalue()
    assert "无消融指纹" in out, out
    print("  ✓ 旧 ckpt（无指纹）只 warn 不崩（向后兼容）")


def test_f5_tokenizer_warn():
    from training.dataline.complete_dataset import CompleteV1Dataset
    CFG = "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # 不传 tokenizer_path（不触发实际数据加载，只看构造期 warn）
        CompleteV1Dataset(sampling_config=CFG, stage="256", image_size=256,
                          max_shards=1, rank=0, world_size=1,
                          stats_cache="/tmp/_nonexist_stats_skip.json") if False else None
        # __init__ 会读 shard/算 stats，太重；直接验证 warn 分支逻辑：tokenizer_path 缺省时打印
        ds_warn = _probe_init_warn(CFG)
    out = buf.getvalue()
    assert "未提供 tokenizer_path" in out, out
    print("  ✓ 无 tokenizer_path → 构造期 warn（退 proxy 截断提示）")


def _probe_init_warn(cfg_path):
    """只跑 __init__ 里 tokenizer 分支的 warn（绕开重的 shard/stats 加载）——
    通过把 list_shards/compute_real_global_stats monkeypatch 成空来隔离。"""
    import training.dataline.complete_dataset as cd
    orig_list, orig_stats = cd.list_shards, cd.compute_real_global_stats
    cd.list_shards = lambda c: {"anime": [], "monet_real": [], "gpic": [], "synthetic": []}
    cd.compute_real_global_stats = lambda *a, **k: {"source_counts": {}, "cluster_sizes": {}}
    try:
        return cd.CompleteV1Dataset(sampling_config=cfg_path, stage="256", image_size=256,
                                    rank=0, world_size=1)
    finally:
        cd.list_shards, cd.compute_real_global_stats = orig_list, orig_stats


if __name__ == "__main__":
    print("[1] F2 指纹 bidirectional"); test_fingerprint_bidirectional()
    print("[2] F2 指纹 其他开关"); test_fingerprint_other_switches()
    print("[3] F2 指纹 旧 ckpt 兼容"); test_fingerprint_legacy_none()
    print("[4] F5 tokenizer warn"); test_f5_tokenizer_warn()
    print("\nALL AUDIT-3 FIX TESTS PASSED")
