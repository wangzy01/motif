#!/usr/bin/env python3
"""Static fail-closed contract for stage512a prep config and JCC templates."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

from training.dataline.config import load_sampling_config
from scripts.offline_arb.make_freeze_manifest import (
    SOURCE_EXPLICIT_STAGE512A,
    STAGE512A_EXCEPTION_PRODUCER_SOURCES,
)
from scripts.offline_arb.render_stage512a_write_job import (
    EXPECTED_BINDING_PATHS,
    EXPECTED_TEMPLATE_PATHS,
    REQUIRED_BINDINGS,
)


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/stage512a_offline_exact_v2.yaml"
JOBS = (
    "offline_arb_phase3a_bstats_stage512a.yaml",
    "offline_arb_phase5_catalog_stage512a.yaml",
    "offline_arb_phase5_plans_stage512a.yaml",
    "offline_arb_animefreq_stage512a.yaml",
    "offline_arb_caption_sidecar_build_stage512a.yaml",
    "offline_arb_exceptions_audit_stage512a.yaml",
    "offline_arb_textatlas_payload_audit_stage512a.yaml",
    "offline_arb_textatlas_payload_finalize_stage512a.yaml",
    "offline_arb_stage512a_reviewed_exception_gate.yaml",
    "offline_arb_freeze_v4_stage512a_reviewed.yaml",
    "offline_arb_stage512a_targeted_exception_preflight.yaml",
    "offline_arb_gpic_reuse_stage512a_producer.yaml",
    "offline_arb_gpic_reuse_stage512a_verify_publish.yaml",
    "offline_arb_stage256_completion_gate_for_stage512a.yaml",
    "offline_arb_stage512a_static_validate.yaml",
    "offline_exact_512a_2x8_GPUVAL_TEMPLATE.yaml",
    "offline_exact_512a_6x8_FORMAL_TEMPLATE.yaml",
)
WRITE_TEMPLATES = (
    "offline_arb_phase5_catalog_stage512a.yaml",
    "offline_arb_phase5_plans_stage512a.yaml",
    "offline_arb_animefreq_stage512a.yaml",
    "offline_arb_caption_sidecar_build_stage512a.yaml",
    "offline_arb_stage512a_reviewed_exception_gate.yaml",
    "offline_arb_freeze_v4_stage512a_reviewed.yaml",
    "offline_arb_gpic_reuse_stage512a_producer.yaml",
    "offline_arb_gpic_reuse_stage512a_verify_publish.yaml",
)
WRITE_PLACEHOLDERS = (
    "__STAGE512A_WRITE_JOB_NAME_UNRENDERED__",
    "__STAGE512A_WRITE_TOKEN_UNRENDERED__",
    "__STAGE512A_WRITE_BINDINGS_B64_UNRENDERED__",
    "__STAGE512A_WRITE_YAML_SELF_UNRENDERED__",
)
WRITE_LOCKS = {
    "offline_arb_phase5_catalog_stage512a.yaml": (
        ".offline_arb_catalog_stage512a_v1.writer.lock",
        'test ! -e "$ROOT"',
    ),
    "offline_arb_phase5_plans_stage512a.yaml": (
        ".offline_arb_plans_stage512a_v1.writer.lock",
        'test ! -e "$ROOT"',
    ),
    "offline_arb_animefreq_stage512a.yaml": (
        ".offline_arb_animefreq_stage512a_v1.writer.lock",
        'test ! -e "$ROOT"',
    ),
    "offline_arb_caption_sidecar_build_stage512a.yaml": (
        ".offline_arb_caption_sidecar_stage512a_v1.writer.lock",
        'test ! -e "$COORD"',
    ),
    "offline_arb_stage512a_reviewed_exception_gate.yaml": (
        ".offline_arb_exceptions_stage512a_reviewed_v1.writer.lock",
        '"$PY" scripts/offline_arb/finalize_stage512a_exception_gate.py',
    ),
    "offline_arb_freeze_v4_stage512a_reviewed.yaml": (
        ".freeze_v4_stage512a_reviewed.lock",
        'test ! -e "$FREEZE"',
    ),
    "offline_arb_gpic_reuse_stage512a_producer.yaml": (
        ".offline_gpic_blob_stage512a_candidate_v1.writer.lock",
        'test ! -e "$OUTPUT"',
    ),
    "offline_arb_gpic_reuse_stage512a_verify_publish.yaml": (
        ".offline_gpic_blob_stage512a_v1.writer.lock",
        'test ! -e "$OUTPUT"',
    ),
}
SAFE_DIRFD_LOCK_TEMPLATES = {
    "offline_arb_stage512a_reviewed_exception_gate.yaml",
    "offline_arb_freeze_v4_stage512a_reviewed.yaml",
}
IDENTITY_KEYS = (
    "stage",
    "image_size",
    "eligibility",
    "requires_nl",
    "anime_nl_filter_applied",
)
LIVE_REPO_RE = re.compile(
    r"/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop(?:/|(?=[\s\"']))"
)
JCC_SAFE_BASE_NAME_MAX = 56
JCC_BASE_NAME_RE = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?\Z")


def main() -> int:
    config_text = CONFIG.read_text(encoding="utf-8")
    assert "TODO(USER DECISION)" not in config_text
    assert "formal candidate" not in config_text.lower()
    cfg = yaml.safe_load(config_text)
    assert cfg["model_cls"] == "networks.model_mot.MoTT2IModel"
    assert cfg["target_device"] == "cuda"
    assert cfg["model"]["patch_size"] == 1
    assert cfg["model"]["text_adapter"]["enabled"] is False
    assert cfg["model"]["text_adapter"]["multi_layer_fusion"]["enabled"] is False
    assert cfg["model"]["text_max_length"] == 512
    assert cfg["model"]["cfg_dropout_prob"] == 0.1
    assert cfg["model"]["text_rich_cfg_dropout_prob"] == 0.0
    dataset = cfg["dataset"]
    assert dataset["store_cls"].endswith("OfflineExactDataset")
    assert dataset["stage"] == "512a" and dataset["image_size"] == 512
    assert float(dataset["crop_v_anchor"]["anime"]) == 0.15
    for key in (
        "offline_plan_root",
        "offline_catalog_root",
        "offline_anime_freq_root",
        "offline_caption_sidecar_root",
    ):
        value = dataset[key]
        assert "stage512a" in value and "offline_arb_plans_v1" not in value, (key, value)
    mar = dataset["multi_ar"]
    assert mar["assignment_mode"] == "offline_exact_v1"
    assert mar["target_area"] == 262144 and mar["divisor"] == 16
    assert mar["warmup_exposure_images"] == 0 and mar["warmup_steps"] == 0
    assert mar["square_only_domains"] == ["synthetic", "textatlas"]
    assert float(mar["bucket_prob_floor"]) == 0.005
    trainer = cfg["trainer"]
    assert trainer["batch_size"] == 32
    assert trainer["max_steps"] == 166667
    assert trainer["max_exposure_images"] == 256000000
    global_batch = 6 * 8 * trainer["batch_size"]
    assert (trainer["max_steps"] - 1) * global_batch < trainer[
        "max_exposure_images"
    ] <= trainer["max_steps"] * global_batch
    assert trainer["checkpoint_keep_every"] == 60000
    assert trainer["checkpoint_steps"] == 3000
    assert trainer["checkpoint_keep_max"] == 4
    assert trainer["checkpoint_keep_format"] == "weights_ema"
    assert trainer["gradient_clip_val"] == 1.0
    assert trainer["seed"] == 42
    assert trainer["amp_dtype"] == "bfloat16"
    assert trainer["ema"] == {"enabled": True, "decay": 0.9999}
    assert trainer["init_load_optimizer"] is True
    assert trainer["init_from"] == (
        "/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/"
        "stage256_offline_exact_v1/last.pt"
    )
    assert cfg["sigma_sampling"] == {
        "dist": "logit_normal",
        "logit_mean": 0.0,
        "logit_std": 1.0,
        "shift": 1.0,
    }
    assert cfg["optimizer"]["name"] == "torch.optim.AdamW"
    optimizer = cfg["optimizer"]["params"]
    assert abs(float(optimizer["lr"]) - 1.732051e-4) < 1e-12
    assert float(optimizer["weight_decay"]) == 0.0
    assert list(optimizer["betas"]) == [0.9, 0.95]
    assert float(optimizer["eps"]) == 1.0e-8
    assert cfg["scheduler"] == {
        "num_warmup_steps": 500,
        "decay": "constant",
    }

    sampling = load_sampling_config(dataset["sampling_config"])
    assert sampling.domain_mix["512a"] == {
        "anime": 0.37,
        "monet_real": 0.40,
        "gpic": 0.13,
        "synthetic": 0.05,
        "textatlas": 0.05,
    }
    assert sampling.anime_caption.route_weights("512a", True) == {
        "rich_nl": 40.0,
        "nl": 30.0,
        "nl_short": 20.0,
        "tag_medium": 10.0,
    }
    no_nl_routes = sampling.anime_caption.route_weights("512a", False)
    no_nl_total = sum(no_nl_routes.values())
    assert no_nl_total > 0
    assert no_nl_routes.get("tag_medium", 0.0) / no_nl_total == 1.0
    assert all(
        value == 0.0
        for key, value in no_nl_routes.items()
        if key != "tag_medium"
    )
    assert sampling.anime_caption.requires_nl("512a") is False

    for name in JOBS:
        path = REPO / "jobs" / name
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text)
        assert isinstance(value, dict) and isinstance(value.get("command"), str)
        job_name = value.get("name")
        assert isinstance(job_name, str) and job_name
        if not job_name.startswith("__"):
            assert len(job_name) <= JCC_SAFE_BASE_NAME_MAX, (path, job_name)
            assert JCC_BASE_NAME_RE.fullmatch(job_name), (path, job_name)
        if int(value.get("gpuPerWorker", 0)) == 0:
            assert int(value.get("cpuPerWorker", 0)) >= 16, path
        assert str(REPO) in text, f"prep repo path missing: {path}"
        if name == "offline_arb_stage256_completion_gate_for_stage512a.yaml":
            assert text.count(
                "LIVE=/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop"
            ) == 1
            assert "LIVE_CONFIG=$LIVE/configs/stage256_offline_exact_v1.yaml" in text
            assert "LIVE_FORMAL=$LIVE/jobs/offline_exact_256_8x8_FORMAL.yaml" in text
            assert (
                "FREEZE_VERIFIER=$REPO/scripts/offline_arb/make_freeze_manifest.py"
                in text
            )
            assert '"$PY" "$FREEZE_VERIFIER" --verify-freeze' in text
            assert '--freeze "$FREEZE" --repo "$LIVE" --plan "$PLAN"' in text
            assert '--config "$LIVE_CONFIG"' in text
            assert '--freeze-verifier-source "$FREEZE_VERIFIER"' in text
            assert '--freeze-verifier-sha256 "$FREEZE_VERIFIER_SHA"' in text
            assert '(cd "$LIVE"' not in text
            assert '$LIVE/scripts/offline_arb/make_freeze_manifest.py' not in text
            assert '--config "$LIVE_CONFIG" --formal-job "$LIVE_FORMAL"' in text
            assert "offline_arb_stage256_completion_for_stage512a_v2" in text
        else:
            assert not LIVE_REPO_RE.search(text), f"live repo leaked into {path}"
        subprocess.run(
            ["bash", "-n"], input=value["command"], text=True, check=True
        )

    assert set(REQUIRED_BINDINGS) == set(WRITE_TEMPLATES)
    assert set(EXPECTED_BINDING_PATHS) == set(WRITE_TEMPLATES)
    assert set(EXPECTED_TEMPLATE_PATHS) == set(WRITE_TEMPLATES)
    assert EXPECTED_TEMPLATE_PATHS == {
        name: REPO / "jobs" / name for name in WRITE_TEMPLATES
    }
    assert REQUIRED_BINDINGS == {
        name: frozenset(EXPECTED_BINDING_PATHS[name]) for name in WRITE_TEMPLATES
    }
    for bindings in EXPECTED_BINDING_PATHS.values():
        for path in bindings.values():
            assert path.is_absolute(), path
    assert REQUIRED_BINDINGS["offline_arb_phase5_plans_stage512a.yaml"] == {
        "config",
        "catalog_header",
        "n_matrix",
        "sampling_config",
    }
    assert set(WRITE_LOCKS) == set(WRITE_TEMPLATES)
    assert len({value[0] for value in WRITE_LOCKS.values()}) == len(WRITE_TEMPLATES)
    for name in WRITE_TEMPLATES:
        path = REPO / "jobs" / name
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text)
        command = value["command"]
        for placeholder in WRITE_PLACEHOLDERS:
            assert text.count(placeholder) == 1, (path, placeholder)
        assert value["name"] == WRITE_PLACEHOLDERS[0], path
        assert (
            'case "$RENDERED_WRITE_TOKEN$RENDERED_WRITE_BINDINGS_B64'
            '$RENDERED_WRITE_YAML_SELF" in'
        ) in command, path
        guard_index = command.index("(*UNRENDERED*)")
        verify_index = command.index("render_stage512a_write_job.py\" verify-runtime")
        assert guard_index < verify_index < command.index('cd "$REPO"'), path
        assert "exit 64" in command[guard_index:verify_index], path
        assert re.search(r"\bassert\b", command) is None, path
        lock_name, first_target_guard = WRITE_LOCKS[name]
        assert command.count(lock_name) == 1, (path, lock_name)
        if name in SAFE_DIRFD_LOCK_TEMPLATES:
            assert "flock -n 9" not in command, path
            assert command.count('hold-lock --lock "$LOCK"') == 1, path
            assert "STAGE512A_SAFE_WRITER_LOCK_HELD" in command, path
            lock_index = command.index('hold-lock --lock "$LOCK"')
        else:
            assert command.count("flock -n 9") == 1, path
            lock_index = command.index("flock -n 9")
        assert verify_index < lock_index < command.index(first_target_guard), path

    reviewed_bindings = REQUIRED_BINDINGS[
        "offline_arb_stage512a_reviewed_exception_gate.yaml"
    ]
    assert {
        "gate_finalizer",
        "gate_evidence_helper",
        "write_renderer",
    }.issubset(reviewed_bindings)
    freeze_bindings = REQUIRED_BINDINGS[
        "offline_arb_freeze_v4_stage512a_reviewed.yaml"
    ]
    assert {"freeze_builder", "freeze_evidence_helper"}.issubset(freeze_bindings)
    freeze_writer = yaml.safe_load(
        (REPO / "jobs/offline_arb_freeze_v4_stage512a_reviewed.yaml").read_text(
            encoding="utf-8"
        )
    )["command"]
    assert "| tee " not in freeze_writer and "\nmv " not in freeze_writer
    assert freeze_writer.count("capture-command") == 2
    assert freeze_writer.count("publish-line") == 1

    static_command = yaml.safe_load(
        (REPO / "jobs/offline_arb_stage512a_static_validate.yaml").read_text(
            encoding="utf-8"
        )
    )["command"]
    execution_anchor = '"$PY" scripts/offline_arb/test_caption_exceptions_v1.py'
    execution_index = static_command.index(execution_anchor)
    compile_block = static_command[:execution_index]
    execution_block = static_command[execution_index:]
    for relative in (
        "scripts/offline_arb/render_stage512a_write_job.py",
        "scripts/offline_arb/test_stage512a_write_job_renderer.py",
        "scripts/offline_arb/test_stage512a_writer_cli_contract.py",
    ):
        assert compile_block.count(relative) == 1, relative
    for relative in (
        "scripts/offline_arb/test_stage512a_write_job_renderer.py",
        "scripts/offline_arb/test_stage512a_writer_cli_contract.py",
    ):
        assert execution_block.count(relative) == 1, relative

    plan_command = yaml.safe_load(
        (REPO / "jobs/offline_arb_phase5_plans_stage512a.yaml").read_text(
            encoding="utf-8"
        )
    )["command"]
    assert 'mv -T "$WORK" "$ROOT"' in plan_command
    for token in (
        "SAMPLING=/fsx/evan/mot_code/data_analysis/configs/t5_sampling_redesign_v1.json",
        "config sampling path differs from the plan sampling authority",
        "catalog sampling path differs from the plan sampling authority",
    ):
        assert token in plan_command, token
    sidecar_command = yaml.safe_load(
        (REPO / "jobs/offline_arb_caption_sidecar_build_stage512a.yaml").read_text(
            encoding="utf-8"
        )
    )["command"]
    for token in (
        'if [ "$NODE_RANK" = 0 ]',
        "writer-lock.ready",
        "writer-lock.release",
        'done.$NODE_RANK',
        "rank0 did not acquire the caption-sidecar writer lock",
        "caption-sidecar writer completion barrier timed out",
    ):
        assert token in sidecar_command, token
    assert sidecar_command.index("flock -n 9") < sidecar_command.index(
        'mkdir -p "$COORD"'
    )

    write_renderer = (
        REPO / "scripts/offline_arb/render_stage512a_write_job.py"
    ).read_text(encoding="utf-8")
    assert "jcc job submit" not in write_renderer.lower()
    assert "offline_exact_512a_6x8_FORMAL_TEMPLATE.yaml" not in REQUIRED_BINDINGS
    assert "offline_exact_512a_6x8_FORMAL_RENDERED.yaml" not in REQUIRED_BINDINGS

    planner_input = (
        REPO / "scripts/offline_arb/make_planner_input.py"
    ).read_text(encoding="utf-8")
    for token in (
        "_stable_sampling_sha256",
        "close_catalog_sampling",
        "sampling config fingerprint changed across planner load",
        "sampling config SHA-256 differs from catalog header",
        "sampling domain_mix differs from catalog header",
    ):
        assert token in planner_input, token
    planner_test = (
        REPO / "scripts/offline_arb/test_stage512a_planner_v2.py"
    ).read_text(encoding="utf-8")
    for token in (
        "bad_sampling_sha",
        "bad_sampling_mix",
        "expected_fingerprint=sampling_fingerprint",
    ):
        assert token in planner_test, token

    formal_job_text = (
        REPO / "jobs/offline_exact_512a_6x8_FORMAL_TEMPLATE.yaml"
    ).read_text(encoding="utf-8")
    assert "NEVER SUBMIT THIS TEMPLATE" in formal_job_text
    assert "only the separate" in formal_job_text
    formal = yaml.safe_load(formal_job_text)
    assert formal["numWorkers"] == 6 and formal["gpuPerWorker"] == 8
    assert formal["cpuPerWorker"] == 80 and formal["memGBPerWorker"] == 900
    assert formal["maxRetries"] == 3
    formal_text = formal["command"]
    for token in (
        "offline_arb_plans_stage512a_v1/w48_prod/plan",
        "offline_arb_catalog_stage512a_v1",
        "offline_arb_caption_sidecar_stage512a_v1",
        "offline_arb_animefreq_stage512a_v1",
        "offline_gpic_blob_stage512a_v1/FORMAL_RELEASE.json",
        "offline_arb_exceptions_stage512a_reviewed_v1/FORMAL_PASS.json",
        "offline_arb_stage256_completion_for_stage512a_v2/FORMAL_PASS.json",
        "stage512a_offline_exact_v2_w48_prod",
        "__STAGE512A_FORMAL_LAUNCH_TOKEN_UNRENDERED__",
        "__STAGE512A_GPUVAL_PASS_UNRENDERED__",
        "__STAGE512A_REVIEWED_FREEZE_UNRENDERED__",
        "__STAGE512A_FORMAL_YAML_SELF_UNRENDERED__",
        "PYTHONOPTIMIZE must be absent for FORMAL launch",
        "stage512a_launch_guard.py",
        "derive-token",
        "source-evidence",
        '--freeze "$FREEZE" --source-proof "$SOURCE_PROOF"',
        '--gpuval-pass "$GPUVAL_PASS"',
        'prepare "${LAUNCH_GUARD_ARGS[@]}"',
        'verify "${LAUNCH_GUARD_ARGS[@]}"',
        'RUN_MANIFEST_SHA',
        'PASS_MANIFEST_SHA',
        'plan.get("total_steps") == 172032',
        'cfg["trainer"].get("max_steps") == 166667',
        "offline_arb_stage512a_reviewed_exception_gate_v1",
        "reviewed_exact_stage512a_exceptions_v1",
        '"monet_empty_prompt_formal_exposures": 19',
        '"textatlas_raw_candidate_formal_exposures": 136',
        '"textatlas_repaired_horizon_exposures": 11',
        "training/dataline_offline/offline_dataset.py",
        "training/dataline_offline/caption_exceptions_v1.py",
        "training/dataline_offline/image_exceptions_v1.py",
        "training/dataline/image_fetch.py",
        "test_stage512a_reviewed_exceptions.py",
        '"$TARGETED_EXCEPTION_PREFLIGHT" --actual-source',
        '--external-binding reviewed_exception_gate="$EXCEPTION_GATE"',
        "OFFLINE_ARB_AUTO_RESUME_RECONCILE=1",
        "NNODES=6",
        "NW_DECL=$(grep -m1 '^numWorkers:'",
        '"$NW_DECL" = "$NNODES"',
        'INIT_SHA=$(sha256sum "$INIT_FROM"',
        '"$INIT_SHA" = "$SOURCE_CHECKPOINT_SHA"',
        '"$LIVE_INIT_BYTES" = "$SOURCE_CHECKPOINT_BYTES"',
        '"$LIVE_INIT_MTIME_NS" = "$SOURCE_CHECKPOINT_MTIME_NS"',
        '"$LIVE_INIT_DEVICE" = "$SOURCE_CHECKPOINT_DEVICE"',
        '"$LIVE_INIT_INODE" = "$SOURCE_CHECKPOINT_INODE"',
        '"$LIVE_INIT_CTIME_NS" = "$SOURCE_CHECKPOINT_CTIME_NS"',
        '--source-checkpoint-device "$SOURCE_CHECKPOINT_DEVICE"',
        '--source-checkpoint-inode "$SOURCE_CHECKPOINT_INODE"',
        '--source-checkpoint-ctime-ns "$SOURCE_CHECKPOINT_CTIME_NS"',
        "stale/future preflight",
        '"$PASS_TS" -le $((NOW + 60))',
        "cachejanitor.",
        "nodemon.",
        'train.py --config "$CFG"',
    ):
        assert token in formal_text, token

    assert re.search(r"\bassert\b", formal_text) is None
    assert formal_text.index("FORMAL template was not rendered/approved") < formal_text.index(
        "mkdir -p /local/evan/cache"
    )
    assert formal_text.index('prepare "${LAUNCH_GUARD_ARGS[@]}"') < formal_text.index(
        '> "$PASS.tmp"'
    )
    assert formal_text.index('verify "${LAUNCH_GUARD_ARGS[@]}"') < formal_text.index(
        'train.py --config "$CFG"'
    )
    targeted_preflight = '"$TARGETED_EXCEPTION_PREFLIGHT" --actual-source'
    assert formal_text.count(targeted_preflight) == 1
    for later in (
        'if [ "$NODE_RANK" = 0 ]; then',
        'prepare "${LAUNCH_GUARD_ARGS[@]}"',
        'train.py --config "$CFG"',
    ):
        assert formal_text.index(targeted_preflight) < formal_text.index(later), later

    launch_guard = (
        REPO / "scripts/offline_arb/stage512a_launch_guard.py"
    ).read_text(encoding="utf-8")
    renderer = (
        REPO / "scripts/offline_arb/render_stage512a_formal_launch.py"
    ).read_text(encoding="utf-8")
    for token in (
        "EXPECTED_PLAN_HORIZON = 172032",
        "EXPECTED_TRAINING_STEPS = 166667",
        "EXPECTED_EXPOSURES = 256000000",
        "EXPECTED_SCHEDULER_WARMUP = 500",
        'EXPECTED_SOURCE_SCHEMA = "stage256_completion_for_stage512a_v2"',
        "_SOURCE_TOP_LEVEL_KEYS",
        "_SOURCE_AUTHORITY_KEYS",
        '"freeze_verifier_source_path"',
        '"freeze_verifier_source_sha256"',
        '"stage256_freeze_verifier_source"',
        "stage256 freeze-verifier source {key} differs from stage512a freeze",
        "_SOURCE_FREEZE_KEYS",
        "_SOURCE_FREEZE_VERIFY_KEYS",
        "source checkpoint inode differs from preflight",
        "stage256 freeze release digest does not match the live authority",
        "stage256 freeze-verification file identity mismatch",
        "checkpoint namespace contains unrecognized crash residue",
        "run manifest bindings differ from current formal inputs",
        'if "PYTHONOPTIMIZE" in os.environ',
        'TEMPLATE_GPUVAL_PASS = "__STAGE512A_GPUVAL_PASS_UNRENDERED__"',
        'TEMPLATE_REVIEWED_FREEZE = "__STAGE512A_REVIEWED_FREEZE_UNRENDERED__"',
        'EXPECTED_STAGE512A_FREEZE_SCHEMA = (',
        '"offline_arb_freeze_v4_stage512a_reviewed_exceptions"',
        '"offline_arb_stage512a_reviewed_exception_gate_v1"',
        '"reviewed_exact_stage512a_exceptions_v1"',
        '"offline_arb_exceptions_stage512a_reviewed_v1/FORMAL_PASS.json"',
    ):
        assert token in launch_guard, token
    for token in (
        'APPROVAL_PHRASE = "STAGE512A_6X8_FORMAL"',
        "render-only-no-submit",
        "publish_exclusive(output, raw, mode=0o444)",
        "derive_launch_token",
        'parser.add_argument("--gpuval-pass", required=True)',
        "TEMPLATE_REVIEWED_FREEZE",
    ):
        assert token in renderer, token
    assert "jcc job submit" not in renderer.lower()

    assert re.search(
        r'--nnodes\s+"\$NNODES"\s+--nproc-per-node\s+8\s+'
        r'--node-rank\s+"\$NODE_RANK"',
        formal_text,
    )
    assert re.search(
        r"while true; do.*?df --output=pcent /local.*?sleep 3600.*?"
        r"cachejanitor\.",
        formal_text,
        re.DOTALL,
    )
    assert re.search(
        r"while true; do.*?memory\.current.*?memory\.pressure.*?"
        r"sleep 120.*?nodemon\.",
        formal_text,
        re.DOTALL,
    )

    verify_start = formal_text.index(
        "scripts/offline_arb/make_freeze_manifest.py --verify-freeze"
    )
    verify_end = formal_text.index('> "$VERIFY.tmp"', verify_start)
    verify_command = formal_text[verify_start:verify_end]
    assert '--config "$CFG"' in verify_command
    assert '--external-binding reviewed_exception_gate="$EXCEPTION_GATE"' in verify_command

    for relative in (
        "scripts/offline_arb/make_planner_input.py",
        "scripts/offline_arb/audit_textatlas_plan_rows.py",
        "scripts/offline_arb/finalize_stage512a_exception_gate.py",
        "scripts/offline_arb/test_stage256_completion_contract.py",
        "scripts/offline_arb/render_stage512a_write_job.py",
        "scripts/offline_arb/test_stage512a_write_job_renderer.py",
        "scripts/offline_arb/test_stage512a_writer_cli_contract.py",
        "scripts/offline_arb/stage512a_launch_guard.py",
        "scripts/offline_arb/render_stage512a_formal_launch.py",
        "scripts/offline_arb/test_stage512a_launch_guard.py",
        "scripts/offline_arb/stage512a_gpuval_guard.py",
        "scripts/offline_arb/render_stage512a_gpuval.py",
        "scripts/offline_arb/test_stage512a_gpuval_contract.py",
    ):
        text = (REPO / relative).read_text(encoding="utf-8")
        assert not LIVE_REPO_RE.search(text), f"live repo leaked into {relative}"

    completion_verifier = (
        REPO / "scripts/offline_arb/verify_stage256_completion_for_stage512a.py"
    ).read_text(encoding="utf-8")
    assert completion_verifier.count(
        '/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop"'
    ) == 1
    assert 'schema": "stage256_completion_for_stage512a_v2"' in completion_verifier
    assert '"freeze_verifier_source_path"' in completion_verifier
    assert '"freeze_verifier_source_sha256"' in completion_verifier

    freeze_verifier = (
        REPO / "scripts/offline_arb/make_freeze_manifest.py"
    ).read_text(encoding="utf-8")
    assert '"scripts/offline_arb/make_freeze_manifest.py"' in freeze_verifier
    for token in (
        "validate_stage512a_catalog_sampling",
        "stage512a catalog sampling SHA-256 differs from current sampling",
        "stage512a catalog domain_mix differs from current sampling",
        "stage512a plan domain_mix differs from current sampling",
        "stage512a plan sampling SHA-256 differs from current sampling",
        "expected_contract, sampling_evidence = load_stage_contract",
    ):
        assert token in freeze_verifier, token
    freeze_profile_test = (
        REPO / "scripts/offline_arb/test_freeze_stage_source_profiles.py"
    ).read_text(encoding="utf-8")
    renderer_relative = "scripts/offline_arb/render_stage512a_write_job.py"
    assert STAGE512A_EXCEPTION_PRODUCER_SOURCES["renderer"] == renderer_relative
    for relative in (
        renderer_relative,
        "scripts/offline_arb/test_stage512a_write_job_renderer.py",
        "scripts/offline_arb/test_stage512a_writer_cli_contract.py",
    ):
        assert SOURCE_EXPLICIT_STAGE512A.count(relative) == 1, relative
        assert freeze_profile_test.count(f'"{relative}"') == 2, relative
    for token in (
        "bad_sampling_sha",
        "bad_sampling_mix",
        "bad_plan_mix",
        "bad_plan_sha",
        '{"stage": "256"}, None',
    ):
        assert token in freeze_profile_test, token

    runbook = (REPO / "docs/STAGE512A_OFFLINE_PREP_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    assert "offline_arb_stage256_completion_for_stage512a_v1" not in runbook
    assert runbook.count(
        "offline_arb_stage256_completion_for_stage512a_v2/FORMAL_PASS.json"
    ) >= 2
    assert "render_stage512a_formal_launch.py" in runbook
    assert "--approve-launch STAGE512A_6X8_FORMAL" in runbook
    assert "GPUVAL_PASS.json" in runbook
    assert '--gpuval-pass "$GPUVAL_PASS"' in runbook
    assert "plans: `config`, `catalog_header`, `n_matrix`, `sampling_config`" in runbook
    assert "stable fingerprint/SHA reads" in runbook
    assert "legacy stage256 freeze contract is unchanged" in runbook
    assert "The launch guard rehashes this verifier source" in runbook
    assert "the proof alone is never trusted" in runbook

    print(
        "STAGE512A_PREP_CONTRACT_PASS "
        f"jobs={len(JOBS)} identity_keys={','.join(IDENTITY_KEYS)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
