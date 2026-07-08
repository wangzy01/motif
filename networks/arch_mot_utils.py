"""无状态 mask / position 纯函数，从 arch_mot.py 抽离（便于 debug 脚本单测/可视化）。

本模块只放**纯函数**：无 self、无模块级可变状态，仅依赖 torch 张量运算，
行为与原 arch_mot.py 中的同名函数逐位一致。arch_mot.py 仍 re-export 这些名字，
故 `from networks.arch_mot import build_t2i_attention_mask` 等旧调用路径不变。
"""

import torch


# --------------------------------------------------------------------------- #
# Mask & position helpers（独立函数便于 debug 脚本单测/可视化）
# --------------------------------------------------------------------------- #
def build_t2i_attention_mask(text_mask: torch.Tensor, num_img_tokens: int) -> torch.Tensor:
    """构造 T2I 联合序列 [text | gen-image] 的 attention mask。

    规则（MoT方案0608「Attention 设计」，t2i 无理解侧干净图）：
      * 文本→文本：causal（沿用 Qwen3-VL 原生），padding 不作为 key；
      * 文本→噪声图：不可见；
      * 噪声图→文本：可见全部有效文本 token（双向语义下文本是条件）；
      * 噪声图→噪声图：bidirectional 全可见。

    Args:
        text_mask: [B, S_t] bool/0-1，有效文本 token 为 True（右 padding）。
        num_img_tokens: 每样本图 token 数（batch 内一致，bucket 保证）。
    Returns:
        [B, 1, S, S] bool，True = 允许 attend（SDPA 语义）。
    """
    bsz, s_t = text_mask.shape
    s_i = num_img_tokens
    s = s_t + s_i
    device = text_mask.device
    valid = text_mask.bool()

    mask = torch.zeros(bsz, s, s, dtype=torch.bool, device=device)
    # 文本块：causal ∧ key 有效。padding 行也给 causal（行输出不被使用，
    # 保留对角线避免全 -inf softmax NaN）。
    causal = torch.tril(torch.ones(s_t, s_t, dtype=torch.bool, device=device))
    mask[:, :s_t, :s_t] = causal.unsqueeze(0) & valid.unsqueeze(1)
    mask[:, :s_t, :s_t] |= torch.eye(s_t, dtype=torch.bool, device=device).unsqueeze(0)
    # 图块行：看有效文本 + 全部图 token。
    mask[:, s_t:, :s_t] = valid.unsqueeze(1).expand(bsz, s_i, s_t)
    mask[:, s_t:, s_t:] = True
    return mask.unsqueeze(1)


def build_t2i_position_ids(
    text_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    time_gap: int = 2048,
) -> torch.Tensor:
    """构造 Interleaved-3D-MRoPE 的 position ids，[3, B, S]（t/h/w 三轴）。

    规则（MoT方案0608「位置编码」）：
      * 文本退化 1D：t=h=w=token 位置（含 padding 位，反正被 mask）；
      * 生成图像：t = T_gen = max(context_t) + Δ（Δ=time_gap=2048，参考 Cosmos3），
        其中 max(context_t) + 1 == 有效文本长度（即 T_gen = text_len - 1 + Δ + 1
        简化为 text_len + Δ - 1 ... 这里取 T_gen = text_len + Δ，语义 =「文本结束
        后空 Δ 个时间步」，每样本随有效长度变化）；
      * 生成侧 h/w 段内从 0 起（绝对段内坐标）。
    """
    bsz, s_t = text_mask.shape
    device = text_mask.device
    s_i = grid_h * grid_w

    pos = torch.zeros(3, bsz, s_t + s_i, dtype=torch.long, device=device)
    text_pos = torch.arange(s_t, device=device).unsqueeze(0).expand(bsz, s_t)
    pos[:, :, :s_t] = text_pos.unsqueeze(0)

    text_len = text_mask.bool().sum(dim=1)  # [B] 有效长度
    t_gen = text_len + time_gap  # [B]
    hh = torch.arange(grid_h, device=device).repeat_interleave(grid_w)  # [S_i]
    ww = torch.arange(grid_w, device=device).repeat(grid_h)  # [S_i]
    pos[0, :, s_t:] = t_gen.unsqueeze(1)
    pos[1, :, s_t:] = hh.unsqueeze(0)
    pos[2, :, s_t:] = ww.unsqueeze(0)
    return pos


def build_2d_sincos_pos_embed(
    grid_h: int, grid_w: int, dim: int, device=None, dtype=None
) -> torch.Tensor:
    """标准 2D sinusoidal 位置编码 [grid_h*grid_w, dim]（h/w 各占一半）。

    用于 B4「绝对 sinusoidal 位置」消融：仅生成 token，AR 分桶下按**当前 grid 现算**
    （不做 i1 那种方图插值）。dim 必须能被 4 整除（h/w 各 dim/2，再 sin/cos 各半）。
    token 顺序与 build_t2i_position_ids 一致（h 外层 repeat_interleave、w 内层 repeat）。
    """
    assert dim % 4 == 0, f"abs-sincos dim 必须能被 4 整除，得到 {dim}"
    quarter = dim // 4
    omega = torch.arange(quarter, dtype=torch.float32, device=device) / quarter
    omega = 1.0 / (10000.0**omega)  # [dim/4]
    hh = torch.arange(grid_h, dtype=torch.float32, device=device).repeat_interleave(grid_w)  # [N]
    ww = torch.arange(grid_w, dtype=torch.float32, device=device).repeat(grid_h)  # [N]
    emb_h = torch.cat([torch.sin(hh[:, None] * omega[None]), torch.cos(hh[:, None] * omega[None])], dim=-1)
    emb_w = torch.cat([torch.sin(ww[:, None] * omega[None]), torch.cos(ww[:, None] * omega[None])], dim=-1)
    emb = torch.cat([emb_h, emb_w], dim=-1)  # [N, dim]
    return emb if dtype is None else emb.to(dtype)
