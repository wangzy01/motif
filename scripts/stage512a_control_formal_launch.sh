#!/usr/bin/env bash
# Runtime command for the externally rendered Stage512a-control FORMAL JCC spec.
set -euo pipefail

fatal() { echo "FATAL: $*" >&2; exit 43; }
need() { [ -n "${!1:-}" ] || fatal "$1 unset"; }
for name in FORMAL_REPO EXPECTED_COMMIT EXPECTED_CONFIG_SHA256 CKPT_DIR RUN_MODE RUN_ID \
            RESOLVED_SPEC_PATH RELEASE_MANIFEST_PATH TRAINING_LINEAGE_PATH \
            TRAINING_LINEAGE_SHA256; do
  need "$name"
done

REPO=$FORMAL_REPO
PY=/fsx/evan/mot_code/venv-gpu/bin/python
PROTO=scripts/stage512a_control_formal_protocol.py
CFG=configs/stage512a_continue_control_rngfix_v1.yaml
NNODES=8
PORT=29817
WAIT_S=${ARRIVAL_WAIT_S:-3600}
EXPECTED_CKPT=/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage512a_continue_control_rngfix_v1
INIT=/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage512a_continue_control_rngfix_v1_inputs/a_final_step125000.last.pt
INIT_SHA=957bfcf12b89c4042a32056566012a2451e72aa0b2706b0ca51808ab79155c31
PLAN=/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop/offline_arb_plan_stage512a_continue_control_rngfix_v1_w64_256m_seed20260710_20260809
SAMPLING=/fsx/evan/mot_code/data_analysis/configs/t5_sampling_redesign_v1.json
CONTRACT=$REPO/provenance/imported_control_scripts/weighted_contract_stage512a_v1.json
COORD=/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/.stage512a_continue_control_rngfix_v1_coord
LOCK=$COORD/active_training.lock

[ -x "$PY" ] || fatal "python runtime missing"
[ "$CKPT_DIR" = "$EXPECTED_CKPT" ] || fatal "checkpoint path is not the frozen control path"
[ "$RUN_ID" = stage512a_continue_control_rngfix_v1 ] || fatal "RUN_ID mismatch"
case "$RUN_MODE" in fresh|continue) ;; *) fatal "RUN_MODE must be fresh or continue";; esac
case "$EXPECTED_COMMIT" in [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*)
  [ "${#EXPECTED_COMMIT}" -eq 40 ] || fatal "EXPECTED_COMMIT must be full SHA";;
  *) fatal "EXPECTED_COMMIT malformed";;
esac
case "$TRAINING_LINEAGE_SHA256" in [0-9a-f][0-9a-f]*)
  [ "${#TRAINING_LINEAGE_SHA256}" -eq 64 ] || fatal "training lineage SHA malformed";;
  *) fatal "training lineage SHA malformed";;
esac

HOST=$(hostname)
NODE_RANK=${HOST##*-}
JOBTAG=${HOST%-*}
case "$NODE_RANK" in (*[!0-9]*|'') fatal "malformed node rank: $NODE_RANK";; esac
[ "$NODE_RANK" -lt "$NNODES" ] || fatal "node rank out of range"

cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONPYCACHEPREFIX=/local/evan/cache/pycache/stage512a_continue_control_rngfix_v1
export HF_HOME=/local/evan/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/local/evan/cache/torch TRITON_CACHE_DIR=/local/evan/cache/triton
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN FI_EFA_USE_DEVICE_RDMA=1
export OFFLINE_ARB_PREFETCH_STEPS=4 OFFLINE_ARB_AUTO_RESUME_RECONCILE=1
unset OFFLINE_ARB_STEP_TIMEOUT_S OFFLINE_ARB_PROFILE_WINDOWS OFFLINE_ARB_SMOKE_DATASET_CONFIG
unset OFFLINE_ARB_IO_TELEMETRY_DIR OFFLINE_ARB_GPIC_THREADS_PER_NODE
unset MALLOC_ARENA_MAX MALLOC_TRIM_THRESHOLD_ ARROW_DEFAULT_MEMORY_POOL LD_PRELOAD
unset PYTHONOPTIMIZE
[ "$("$PY" -c 'import sys; print(sys.flags.optimize)')" = 0 ] || fatal "optimized Python forbidden"
mkdir -p /local/evan/anime_img_cache /local/evan/cache "$PYTHONPYCACHEPREFIX"

# Every node binds the immutable code and resolved submission before joining the
# generation handshake.  The resolved spec lives outside the tracked commit,
# avoiding the impossible EXPECTED_COMMIT self-reference.
bash scripts/jcc_preflight.sh "$EXPECTED_COMMIT" "$REPO"
[ "$(sha256sum "$CFG" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256" ] \
  || fatal "config SHA differs from resolved release"
"$PY" "$PROTO" verify-release \
  --release-manifest "$RELEASE_MANIFEST_PATH" --repo "$REPO" \
  --commit "$EXPECTED_COMMIT" --resolved-spec "$RESOLVED_SPEC_PATH" \
  --run-id "$RUN_ID" --checkpoint-dir "$CKPT_DIR"
[ "$(sha256sum "$TRAINING_LINEAGE_PATH" | awk '{print $1}')" = "$TRAINING_LINEAGE_SHA256" ] \
  || fatal "training lineage SHA drift"
"$PY" scripts/stage512a_control_formal_preflight.py \
  --config "$CFG" --contract "$CONTRACT" --release-manifest "$RELEASE_MANIFEST_PATH"

mkdir -p "$COORD"
NODE_NONCE=$("$PY" "$PROTO" token)

COMMON=(--release-manifest "$RELEASE_MANIFEST_PATH" --repo "$REPO"
        --commit "$EXPECTED_COMMIT" --resolved-spec "$RESOLVED_SPEC_PATH"
        --run-id "$RUN_ID" --checkpoint-dir "$CKPT_DIR")

ARCHIVE_RECEIPT=-
if [ "$NODE_RANK" = 0 ]; then
  # Acquired before any checkpoint claim, mode resolution or residual archival,
  # and intentionally inherited by torchrun for its entire lifetime.
  exec 9>"$LOCK"
  flock -n 9 || fatal "another submission already owns the W64 training lease"

  PROBED=$("$PY" "$PROTO" probe-mode "${COMMON[@]}" --run-mode "$RUN_MODE")
  if [ "$PROBED" = fresh ]; then
    [ "$(sha256sum "$INIT" | awk '{print $1}')" = "$INIT_SHA" ] \
      || fatal "A-final init alias full SHA drift"
  fi

  IFS=$'\t' read -r GENERATION GEN_DIR < <(
    "$PY" "$PROTO" mint-generation --coord-root "$COORD" --run-id "$RUN_ID" \
      --jobtag "$JOBTAG" --commit "$EXPECTED_COMMIT" \
      --config-sha "$EXPECTED_CONFIG_SHA256" --node-nonce "$NODE_NONCE")
  IFS=$'\t' read -r EFFECTIVE_MODE ARCHIVE_RECEIPT < <(
    "$PY" "$PROTO" resolve-mode "${COMMON[@]}" --run-mode "$RUN_MODE" \
      --jobtag "$JOBTAG" --generation "$GENERATION")
  "$PY" "$PROTO" write-hello --generation-dir "$GEN_DIR" --rank 0 \
    --node-nonce "$NODE_NONCE" --run-id "$RUN_ID" --jobtag "$JOBTAG" \
    --generation "$GENERATION"
  MASTER_ADDR=$(hostname -i | awk '{print $1}')
  RDZV_JSON=$("$PY" "$PROTO" collect-rdzv --generation-dir "$GEN_DIR" \
    --world-nodes "$NNODES" --run-id "$RUN_ID" --jobtag "$JOBTAG" \
    --generation "$GENERATION" --commit "$EXPECTED_COMMIT" \
    --config-sha "$EXPECTED_CONFIG_SHA256" --mode "$EFFECTIVE_MODE" \
    --master-addr "$MASTER_ADDR" --lineage-sha "$TRAINING_LINEAGE_SHA256" \
    --timeout-s "$WAIT_S")
else
  RDZV_JSON=$("$PY" "$PROTO" await-rdzv --coord-root "$COORD" \
    --rank "$NODE_RANK" --node-nonce "$NODE_NONCE" --run-id "$RUN_ID" \
    --jobtag "$JOBTAG" --commit "$EXPECTED_COMMIT" \
    --config-sha "$EXPECTED_CONFIG_SHA256" --lineage-sha "$TRAINING_LINEAGE_SHA256" \
    --world-nodes "$NNODES" --timeout-s "$WAIT_S")
fi

read -r GENERATION MASTER_ADDR EFFECTIVE_MODE RECONCILE_NONCE ATTEMPT_TAG < <(
  printf '%s' "$RDZV_JSON" | "$PY" -c \
    'import json,sys; d=json.load(sys.stdin); print(d["generation"], d["master_addr"], d["effective_mode"], d["reconcile_nonce"], d["attempt_tag"])')
case "$EFFECTIVE_MODE" in fresh|continue) ;; *) fatal "invalid effective mode";; esac
echo "all 8 nodes arrived generation=$GENERATION with exact nonce map"
echo "run mode: declared=$RUN_MODE effective=$EFFECTIVE_MODE"

# Reconciler identity is common across all nodes/ranks.  Node-local nonces exist
# only in the handshake and never enter the run-stable RNG lineage.
export OFFLINE_ARB_ATTEMPT_JOBTAG="$ATTEMPT_TAG"
export OFFLINE_ARB_ATTEMPT_NONCE="$RECONCILE_NONCE"
export OFFLINE_ARB_AUTO_RESUME_FREEZE_SHA256
OFFLINE_ARB_AUTO_RESUME_FREEZE_SHA256=$(sha256sum "$SAMPLING" | awk '{print $1}')
export OFFLINE_ARB_AUTO_RESUME_PLAN_HEADER_SHA256
OFFLINE_ARB_AUTO_RESUME_PLAN_HEADER_SHA256=$(sha256sum "$PLAN/header.json" | awk '{print $1}')
export MOT_EXPECTED_BOOT_MODE="$EFFECTIVE_MODE"
export MOT_EXPECTED_RUN_LINEAGE_SHA256="$TRAINING_LINEAGE_SHA256"

if [ "$NODE_RANK" = 0 ]; then
  "$PY" "$PROTO" append-receipt "${COMMON[@]}" --mode "$EFFECTIVE_MODE" \
    --jobtag "$JOBTAG" --generation "$GENERATION" \
    --reconcile-nonce "$RECONCILE_NONCE" --archive-receipt "$ARCHIVE_RECEIPT"
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
echo "===== STAGE512A-CONTROL FORMAL node=$NODE_RANK master=$MASTER_ADDR world=64 mode=$EFFECTIVE_MODE gen=$GENERATION ====="
{ hostname; nvidia-smi -L; df -h /local | tail -1; sha256sum "$CFG"; } \
  2>&1 | tee "$CKPT_DIR/probe.n$NODE_RANK.$GENERATION.log"

MONITOR_PIDS=()
cleanup_monitors() {
  for pid in "${MONITOR_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${MONITOR_PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup_monitors EXIT
( exec 9>&-; while true; do
    date -u
    [ -r /sys/fs/cgroup/memory.current ] && echo "mem.current: $(cat /sys/fs/cgroup/memory.current)"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    df --output=avail /local | tail -1
    sleep 300
  done ) > "$CKPT_DIR/nodemon.n$NODE_RANK.$GENERATION.log" 2>&1 &
MONITOR_PIDS+=("$!")
( exec 9>&-; while true; do
    use=$(df --output=pcent /local 2>/dev/null | tail -1 | tr -dc '0-9' || true)
    if [ "${use:-0}" -ge 85 ]; then
      cache_old=/local/evan/anime_img_cache.old.$(date +%s)
      mv /local/evan/anime_img_cache "$cache_old" 2>/dev/null || true
      mkdir -p /local/evan/anime_img_cache
      nice -n 19 rm -rf "$cache_old" &
      echo "$(date -u) rotated anime image cache at ${use}%"
    fi
    sleep 3600
  done ) > "$CKPT_DIR/cachejanitor.n$NODE_RANK.$GENERATION.log" 2>&1 &
MONITOR_PIDS+=("$!")

LOG=$CKPT_DIR/train.n$NODE_RANK.$GENERATION.log
set +e
"$PY" -m torch.distributed.run \
  --nnodes "$NNODES" --nproc-per-node 8 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$PORT" \
  train.py --config "$CFG" trainer.checkpoint_dir="$CKPT_DIR" 2>&1 | tee "$LOG"
PIPE_RC=("${PIPESTATUS[@]}")
set -e
[ "${PIPE_RC[1]}" -eq 0 ] || exit "${PIPE_RC[1]}"
RC=${PIPE_RC[0]}

if [ "$NODE_RANK" = 0 ]; then
  grep -Fq "[formal-boot] verified mode=$EFFECTIVE_MODE" "$LOG" \
    || fatal "formal boot marker missing"
  if [ "$EFFECTIVE_MODE" = fresh ]; then
    grep -Fq "[train] init_from $INIT (load_optimizer=True); step" "$LOG" \
      || fatal "fresh boot lacks exact A-final init receipt"
    grep -Fq "[rng] init_from legacy_missing_rng accepted" "$LOG" \
      || fatal "fresh boot lacks legacy-to-new RNG receipt"
    ! grep -q "resumed from .* at step [1-9]" "$LOG" \
      || fatal "fresh boot unexpectedly resumed local progress"
  else
    grep -q "resumed from .* at step [1-9]" "$LOG" \
      || fatal "continue boot did not load local full state"
    grep -Fq "[rng] restored checkpoint RNG schema=mot_rng_state_v1 rank=0 world_size=64" "$LOG" \
      || fatal "continue boot did not restore rank RNG"
    grep -Fq "[auto-resume] reconciliation=" "$LOG" \
      || fatal "continue boot lacks reconciliation receipt"
  fi
fi
[ "$RC" -eq 0 ] || { echo "training exited $RC"; exit "$RC"; }

if [ "$NODE_RANK" = 0 ]; then
  "$PY" - "$CKPT_DIR" <<'PY'
import pathlib, sys, torch
root = pathlib.Path(sys.argv[1])
for step in (30000, 60000, 90000, 120000):
    path = root / f"step_{step}.pt"
    assert path.is_file() and not path.is_symlink() and path.stat().st_size > 20_000_000_000
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert value["checkpoint_schema"] == "mot_full_training_v2"
    assert int(value["global_step"]) == step
    assert value["rng_state"]["schema"] == "mot_rng_state_v1"
    assert "optimizer" in value and "ema" in value
    del value
print("permanent full-state snapshots verified: 30000 60000 90000 120000")
PY
fi
echo "=== DONE ==="
