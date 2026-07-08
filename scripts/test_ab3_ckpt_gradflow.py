"""Ab3 und_lora grad-flow 复核：gradient_checkpointing=True（与产线一致）下，
验证 LoRA up/down 拿到梯度、frozen und base 无梯度。

既有 scripts/test_arch_changes.py::test_und_lora_baseline_joint 在 checkpointing=False 下已过；
本脚本补 checkpointing=True 这一维（torch.utils.checkpoint use_reentrant=False 重算 joint 层时，
LoRA 这类「只 LoRA 参数 require_grad、und base 冻结」的旁路能否正确回传梯度）。

跑法（JCC 单卡）：PYTHONPATH=<worktree> python scripts/test_ab3_ckpt_gradflow.py
通过 = 末行打印 "ALL OK"，退出码 0。
"""
import torch

from scripts.test_arch_changes import build_transformer, sample_inputs, randomize_proj_out, fwd


def _nonzero_grad(params):
    return any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in params)


def run(device):
    lora_cfg = dict(enabled=True, num_layers=2, rank=4, alpha=4,
                    target_modules=["q", "k", "v", "o"])
    m = build_transformer(seed=0, und_lora=lora_cfg)
    randomize_proj_out(m)
    m.enable_gradient_checkpointing()      # ★ 与产线 config(gradient_checkpointing=true) 一致
    assert m.gradient_checkpointing is True, "enable_gradient_checkpointing 未生效"
    m.freeze_understanding()
    m = m.to(device)
    m.train()                              # checkpoint 仅在 self.training 时触发（arch_mot 917/1071）

    inp = tuple(x.to(device) if torch.is_tensor(x) else x for x in sample_inputs())

    up = [p for n, p in m.named_parameters() if "lora_und.up.weight" in n]
    down = [p for n, p in m.named_parameters() if "lora_und.down.weight" in n]
    assert up and down, f"未建出 LoRA up/down 参数: up={len(up)} down={len(down)}"

    # (1) up zero-init：up 应拿到非零梯度；down 此时梯度应为 0（∂/∂down ∝ up=0）。
    fwd(m, inp).pow(2).mean().backward()
    assert _nonzero_grad(up), "checkpointing=True 下 LoRA up 未拿到非零梯度（grad 流断了）"
    assert not _nonzero_grad(down), "up=0 时 down 不应有非零梯度"

    # (2) frozen und base：不可有任何梯度。
    frozen_with_grad = [n for n, p in m.named_parameters()
                        if not m.is_trainable_param_name(n) and p.grad is not None]
    assert not frozen_with_grad, f"冻结 und 参数拿到梯度: {frozen_with_grad[:5]}"

    # (3) up 置非零后：down 应拿到非零梯度（完整 A/B 都能学）。
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        for p in up:
            p.fill_(0.05)
    fwd(m, inp).pow(2).mean().backward()
    assert _nonzero_grad(down), "up≠0 后 LoRA down 仍未拿到非零梯度"
    assert not [n for n, p in m.named_parameters()
                if not m.is_trainable_param_name(n) and p.grad is not None], "二次 backward 冻结参数有梯度"

    print(f"[ab3-ckpt-gradflow] device={device} OK: "
          f"up_grad✓ down_grad(after up!=0)✓ frozen_und_no_grad✓  (gradient_checkpointing=True)")


def main():
    run(torch.device("cpu"))
    if torch.cuda.is_available():
        run(torch.device("cuda:0"))
    print("ALL OK")


if __name__ == "__main__":
    main()
