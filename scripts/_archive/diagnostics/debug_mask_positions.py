#!/usr/bin/env python3
"""可视化 + 断言检查：attention mask 与 MRoPE position ids 的构造逻辑。

用法（项目根目录）: venv/bin/python scripts/debug_mask_positions.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from networks.arch_mot import build_t2i_attention_mask, build_t2i_position_ids


def main():
    # 2 个样本：文本有效长度 4 / 6，pad 到 8；图 3x2=6 token
    s_t, grid_h, grid_w = 8, 3, 2
    text_mask = torch.zeros(2, s_t, dtype=torch.bool)
    text_mask[0, :4] = True
    text_mask[1, :6] = True

    mask = build_t2i_attention_mask(text_mask, grid_h * grid_w)
    pos = build_t2i_position_ids(text_mask, grid_h, grid_w, time_gap=2048)
    s = s_t + grid_h * grid_w

    print(f"mask shape: {tuple(mask.shape)}  (期望 [2, 1, {s}, {s}])")
    print(f"pos  shape: {tuple(pos.shape)}   (期望 [3, 2, {s}])")

    for b in range(2):
        n_valid = int(text_mask[b].sum())
        print(f"\n===== 样本 {b}（有效文本 {n_valid}, pad {s_t - n_valid}, 图 {grid_h}x{grid_w}）=====")
        print("attention mask（行=query，列=key；#=可见，.=不可见；| 分隔文本/图）：")
        m = mask[b, 0]
        for i in range(s):
            row = "".join("#" if m[i, j] else "." for j in range(s_t)) + "|" + \
                  "".join("#" if m[i, j + s_t] else "." for j in range(grid_h * grid_w))
            tag = "txt" if i < s_t else "img"
            pad = " (pad)" if i < s_t and not text_mask[b, i] else ""
            print(f"  {tag}{i:02d} {row}{pad}")

        print("position ids (t/h/w)：")
        for name, axis in zip("thw", range(3)):
            print(f"  {name}: {pos[axis, b].tolist()}")

        # ----- 断言 ----- #
        valid = text_mask[b]
        img_rows = m[s_t:, :]
        # 1. 文本（含pad）看不到任何图 token
        assert not m[:s_t, s_t:].any(), "文本不应看到噪声图"
        # 2. 图 token 可见全部有效文本、不可见 pad
        assert img_rows[:, :s_t][:, valid].all(), "图应看到全部有效文本"
        assert not img_rows[:, :s_t][:, ~valid].any(), "图不应看到 padding"
        # 3. 图内部全可见（bidirectional）
        assert img_rows[:, s_t:].all(), "图内部应 bidirectional 全可见"
        # 4. 有效文本 causal：上三角不可见
        for i in range(n_valid):
            assert m[i, :i + 1].all() and not m[i, i + 1:s_t].any(), f"文本行 {i} 非 causal"
        # 5. 位置：文本 t=h=w 同步递增；图 t = 有效长度 + 2048，h/w 段内从 0 起
        assert (pos[0, b, :s_t] == torch.arange(s_t)).all()
        assert (pos[0, b, :s_t] == pos[1, b, :s_t]).all() and (pos[0, b, :s_t] == pos[2, b, :s_t]).all()
        assert (pos[0, b, s_t:] == n_valid + 2048).all(), "图 t 应为 text_len + Δ"
        assert pos[1, b, s_t:].max() == grid_h - 1 and pos[2, b, s_t:].max() == grid_w - 1
        assert pos[1, b, s_t] == 0 and pos[2, b, s_t] == 0, "图 h/w 应段内从 0 起"
        print("  ✓ 全部断言通过")

    print("\n所有 mask / position 检查通过 ✅")


if __name__ == "__main__":
    main()
