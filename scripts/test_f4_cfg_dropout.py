"""验证 F4 修复（CPU）：CFG dropout 改模型单点 + tag-only anime 按 cfg_keep 豁免。
  - 模型 _apply_cfg_dropout：cfg_keep=True 的样本永不 drop；其余按 cfg_dropout_prob。
  - build_anime_caption 不再自行置空（loader 不重复 dropout）。
  - dataset._is_cfg_protected：anime tag-only=True、anime NL/real/synth=False。
运行：venv-gpu/bin/python scripts/test_f4_cfg_dropout.py
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from training.dataline.config import load_sampling_config
from training.dataline.caption_build import build_anime_caption
from training.dataline.complete_dataset import CompleteV1Dataset

CFG = "/fsx/evan/mot_code/data_analysis/configs/t5_sampling_complete_v1.json"


def test_model_dropout_keep():
    # 用最小 stub 直接测 _apply_cfg_dropout（不建全模型，借未绑定函数）
    class M:
        training = True
        cfg_dropout_prob = 1.0           # 强制 drop 以放大效果
        text_rich_cfg_dropout_prob = 0.0
        dropout_generator = torch.Generator().manual_seed(0)
        _apply_cfg_dropout = CompleteV1Dataset.__dict__.get("_noop")  # placeholder
    from networks.model_mot import MoTT2IModel
    fn = MoTT2IModel._apply_cfg_dropout
    m = M()
    prompts = ["a", "b", "c", "d"]
    # 全 prob=1 且无 keep → 全置空
    out = fn(m, list(prompts))
    assert out == ["", "", "", ""], out
    # cfg_keep=[T,F,T,F] → keep 的位永不置空
    m.dropout_generator = torch.Generator().manual_seed(0)
    out = fn(m, list(prompts), None, [True, False, True, False])
    assert out[0] == "a" and out[2] == "c", out
    assert out[1] == "" and out[3] == "", out
    print("  ✓ 模型 _apply_cfg_dropout：cfg_keep=True 永不 drop、其余按 prob drop")


def test_build_anime_no_dropout():
    cfg = load_sampling_config(CFG)
    rng = random.Random(0)
    f = {"refined_description": "A girl.", "medium_description": "", "short_description": "",
         "rating_bucket": "g", "year": "2021", "tags_general": ["1girl", "solo"],
         "tags_artist": ["x"], "tags_character": ["c"], "tags_copyright": ["s"], "tags_meta": ["highres"]}
    # 跑很多次，build_anime_caption 不应再产生空串（dropout 已移走）
    empties = sum(1 for _ in range(500) if build_anime_caption(f, "256", cfg.anime_caption, rng, {"1girl": 9000}) == "")
    assert empties == 0, f"build_anime_caption 不应再置空，实际空 {empties}"
    print("  ✓ build_anime_caption 不再自行 dropout（0 空串 / 500 次）")


def test_is_cfg_protected():
    nl_rec = {"cap_fields": {"refined_description": "x", "medium_description": "", "short_description": ""}}
    tag_rec = {"cap_fields": {"refined_description": "", "medium_description": "", "short_description": ""}}
    assert CompleteV1Dataset._is_cfg_protected("anime", tag_rec) is True, "anime tag-only 应豁免"
    assert CompleteV1Dataset._is_cfg_protected("anime", nl_rec) is False, "anime NL 不豁免"
    assert CompleteV1Dataset._is_cfg_protected("monet_real", tag_rec) is False, "real 不豁免"
    assert CompleteV1Dataset._is_cfg_protected("synthetic", tag_rec) is False, "synth 不豁免"
    print("  ✓ _is_cfg_protected：仅 anime tag-only=True")


if __name__ == "__main__":
    print("[1] 模型单点 dropout + keep 豁免"); test_model_dropout_keep()
    print("[2] build_anime_caption 不再 dropout"); test_build_anime_no_dropout()
    print("[3] _is_cfg_protected 判定"); test_is_cfg_protected()
    print("\nALL F4 TESTS PASSED")
