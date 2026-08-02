#!/usr/bin/env python3
"""Prove every prompt-audit check can actually fail.

The smoke this exercises has a specific history: its first version read two keys the
loader does not emit, so every check silently no-opped and it reported PASS. A check
that cannot fail is worse than no check, because it is counted as evidence.

So this does not test the audit's happy path (though it covers that too) -- it feeds a
deliberately broken batch to each check in turn and asserts the failure is raised. If
someone refactors PromptAudit and a condition stops matching, the corresponding case
here goes from CAUGHT to MISSED and this file fails, instead of the smoke going quietly
green on a broken loader.

Synthetic batches, real tokenizer. The dataset is a stub carrying only ``last_item_trace``
because that is the entire surface PromptAudit reads.

New tests live under scripts/, not tests/ -- tests/ has an exact-two closure check.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKENIZER_PATH = "/fsx/models/Qwen3-VL-2B-Instruct"
CHECKS = 0
FAILURES: list[str] = []


def check(condition, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL  {label}")


def load_smoke():
    path = ROOT / "scripts/offline_arb/smoke_anchors_loader_v1.py"
    spec = importlib.util.spec_from_file_location("smoke_anchors_loader_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGT:
    def __init__(self, mapping):
        self.mapping = mapping

    def get(self, row_id):
        return self.mapping.get(row_id)


def make_batch(prompts, domains, rows, bands, keeps):
    dataset = types.SimpleNamespace(
        last_item_trace={"domains": domains, "item_rows": rows, "caption_bands": bands}
    )
    batch = {
        "prompts": prompts,
        "cfg_keep": keeps,
        "pixels": torch.zeros(len(prompts), 3, 8, 8),
    }
    return dataset, batch


def main() -> int:
    from transformers import AutoTokenizer

    from networks.model_mot import DEFAULT_PROMPT_TEMPLATE as TEMPLATE

    smoke = load_smoke()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print("happy path")
    audit = smoke.PromptAudit(tokenizer, TEMPLATE, 512, FakeGT({10: "GRAND OPENING"}))
    dataset, batch = make_batch(
        [
            'The image renders the exact text: "GRAND OPENING". A shop window.',
            "a photograph of a dog on a beach",
            "1girl, solo, long hair, school uniform",
            "a synthetic render of a teapot",
            "a busy city street at night",
        ],
        ["textatlas", "monet_real", "anime", "synthetic", "gpic"],
        [10, 20, 30, 40, 50],
        [-1, 2, -1, -1, 0],
        [False, False, True, False, False],
    )
    audit.add(dataset, batch)
    check(audit.finish() == [], "a correct batch produces no failures")
    summary = audit.summary()
    check(summary["rows"] == 5, "all rows counted")
    check(summary["token_length"]["would_truncate"] == 0, "nothing truncates")
    check(summary["cfg_keep_true"] == 1 and summary["cfg_keep_non_anime"] == 0,
          "cfg_keep counted and attributed")
    check(summary["textatlas"] == {"rows": 1, "gt_available": 1, "gt_in_prompt": 1},
          "GT hit recorded")

    print("each check fires on its own fault")

    def fires(label, prompts, domains, rows, bands, keeps, gt=None):
        audit = smoke.PromptAudit(tokenizer, TEMPLATE, 512, FakeGT(gt or {}))
        dataset, batch = make_batch(prompts, domains, rows, bands, keeps)
        audit.add(dataset, batch)
        # Coverage complaints fire on every one-row batch by construction; the point
        # here is whether the SPECIFIC fault was caught.
        noise = ("never sampled", "no textatlas rows sampled", "not one had GT")
        real = [f for f in audit.finish() if not any(n in f for n in noise)]
        check(bool(real), label)
        if real:
            print(f"  ok    {label}\n          {real[0][:88]}")

    fires("cfg_keep on a non-anime row is caught",
          ["p"], ["gpic"], [1], [0], [True])
    fires("band -1 on a real domain is caught",
          ["p"], ["gpic"], [1], [-1], [False])
    fires("a band on textatlas is caught",
          ["p"], ["textatlas"], [1], [2], [False])
    fires("GT present but missing from the prompt is caught",
          ["a photograph"], ["textatlas"], [10], [-1], [False], {10: "GRAND OPENING"})
    fires("a prompt over text_max_length is caught",
          ["word " * 900], ["anime"], [1], [-1], [False])
    fires("an all-empty domain is caught",
          [""], ["anime"], [1], [-1], [False])
    fires("component length disagreement is caught",
          ["a", "b"], ["anime"], [1], [-1], [False])

    print("structural absences fail loudly rather than no-opping")
    audit = smoke.PromptAudit(tokenizer, TEMPLATE, 512, None)
    audit.add(types.SimpleNamespace(last_item_trace=None),
              {"prompts": ["x"], "cfg_keep": [False], "pixels": torch.zeros(1, 3, 8, 8)})
    check(bool(audit.failures), "a missing item trace is caught")

    audit = smoke.PromptAudit(tokenizer, TEMPLATE, 512, None)
    dataset, batch = make_batch(["x"], ["anime"], [1], [-1], [False])
    del batch["cfg_keep"]
    audit.add(dataset, batch)
    check(bool(audit.failures), "a batch without cfg_keep is caught")

    audit = smoke.PromptAudit(tokenizer, TEMPLATE, 512, None)
    check(any("no rows at all" in f for f in audit.finish()),
          "an audit that saw nothing fails instead of passing empty")

    # The GT-coverage assertion is the one most likely to rot into a no-op, because it
    # only means anything when textatlas rows were actually drawn.
    audit = smoke.PromptAudit(tokenizer, TEMPLATE, 512, FakeGT({}))
    dataset, batch = make_batch(["p"], ["textatlas"], [999], [-1], [False])
    audit.add(dataset, batch)
    check(any("not one had GT" in f for f in audit.finish()),
          "textatlas rows with zero GT hits is reported as a binding failure")

    print(f"\n{CHECKS} checks, {len(FAILURES)} failures")
    for failure in FAILURES:
        print(f"  {failure}")
    if FAILURES:
        return 1
    print("SMOKE_PROMPT_AUDIT_TESTS_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
