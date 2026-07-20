#!/usr/bin/env bash
set -euxo pipefail

MODE=${1:-}
case "$MODE" in
  recover384|main) ;;
  *) echo "FATAL: usage: $0 {recover384|main}" >&2; exit 64 ;;
esac

cd /fsx/evan/mot_code/mot_pilot_v1_cleanup_crop
export HF_HOME=/local/evan/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/local/evan/cache/torch TRITON_CACHE_DIR=/local/evan/cache/triton
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
export PYTHONPATH=/fsx/evan/mot_code/mot_pilot_v1_cleanup_crop
CACHE_TAG=${JCC_JOB_ID:-${JOB_ID:-manual}}
export PYTHONPYCACHEPREFIX="/local/evan/stage256_20260718_pycache/$CACHE_TAG/$(hostname)"
mkdir -p "$PYTHONPYCACHEPREFIX"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN FI_EFA_USE_DEVICE_RDMA=1
export OFFLINE_ARB_PREFETCH_STEPS=4
unset OFFLINE_ARB_STEP_TIMEOUT_S OFFLINE_ARB_PROFILE_WINDOWS OFFLINE_ARB_SMOKE_DATASET_CONFIG OFFLINE_ARB_IO_TELEMETRY_DIR
unset OFFLINE_ARB_GPIC_THREADS_PER_NODE
unset MALLOC_ARENA_MAX MALLOC_TRIM_THRESHOLD_ ARROW_DEFAULT_MEMORY_POOL LD_PRELOAD TCMALLOC_RELEASE_RATE
mkdir -p /local/evan/cache/huggingface /local/evan/cache/torch /local/evan/cache/triton /local/evan/anime_img_cache
PY=/fsx/evan/mot_code/venv-gpu/bin/python

RUN_ID=stage256_20260718_causalmlf_offline_exact_v2
BASE=/fsx/evan/mot_code/data_analysis/outputs/mot_pilot_v1_cleanup_crop
ROOT=$BASE/offline_arb_stage256_20260718_semantic_v1
RES=$BASE/$RUN_ID
CK=/fsx/evan/checkpoints/mot_pilot_v1_cleanup_crop/stage256_20260718_causalmlf_offline_exact_v2
CFG=configs/stage256_20260718_causalmlf_offline_exact_v1.yaml
PLAN=$ROOT/w64_prod/plan
CAT=$BASE/offline_arb_catalog_v1
FREEZE=$ROOT/FREEZE_MANIFEST.json
FINAL_HEALTH=$ROOT/FINAL_HEALTH_SUCCESS.json
STATIC=$ROOT/STATIC_SUCCESS.json
STATIC_CONTRACT=$ROOT/STATIC_CONTRACT_SUCCESS.json
PLAN_AUDIT=$PLAN/PLAN_AUDIT_SUCCESS.json
OVERLAY_VERIFY=$ROOT/OVERLAY_VERIFY_SUCCESS.json
V1_YAML=jobs/offline_exact_stage256_20260718_FORMAL_8x8.yaml
V2_YAML=jobs/offline_exact_stage256_20260718_FORMAL_RESUME_8x8_v2.yaml
ADDENDUM=docs/FORMAL_V3_RECOVERY_ADDENDUM.md
RUNNER=scripts/offline_arb/run_stage256_20260718_formal_v3.sh
BOOT_RECEIPT=$ROOT/formal_v2/BOOT256_CHECKPOINT_SUCCESS.json
GPIC_PREFLIGHT=scripts/offline_arb/verify_offline_gpic_compact_controls.py
ARTIFACT_BUILDTAG=evan-mot-crop-offlinearb-gpic-s3-production-build-zg8g
ARTIFACT_ROOT=$BASE/offline_gpic_blob_v1_release/$ARTIFACT_BUILDTAG
MANIFEST_SHA256=a18ba1f1250e3b00abdc1051dc71e08a254b8855d5b61b9975e88d2e6ae2ee05
EXPECTED_ARCH_FP_SHA256=70546507a768a9d01705b8118b6e3cfea3f055f22db5e9bcdfb3e25e7d022642
EXPECTED_FINAL_HEALTH_SHA256=7137e2f15811d24162e579e1538b7e3bdb6bffb41fb2e8d3b1b63e2d82040c6d
EXACT_EXPOSURES=384000768
NNODES=8

if [ "$MODE" = recover384 ]; then
  YAML_SELF=jobs/offline_exact_stage256_20260718_FORMAL_RECOVER384_8x8_v3.yaml
  RECEIPT=$ROOT/formal_v3/PRE_RECOVER256_SUCCESS.json
  RECEIPT_SCHEMA=stage256_20260718_formal_v3_pre_recover256_v1
  EXPECTED_CHECKPOINT_STEP=256
  EXPECTED_METRICS_STEP=430
  ATTEMPT_PARENT=$ROOT/formal_v3_recover384_attempts
  PORT=29795
  CLI_BOUNDARY=(--max-steps 384)
else
  YAML_SELF=jobs/offline_exact_stage256_20260718_FORMAL_RESUME_8x8_v3.yaml
  RECEIPT=$ROOT/formal_v3/RECOVER384_CHECKPOINT_SUCCESS.json
  RECEIPT_SCHEMA=stage256_20260718_formal_v3_recover384_checkpoint_v1
  EXPECTED_CHECKPOINT_STEP=384
  EXPECTED_METRICS_STEP=384
  ATTEMPT_PARENT=$ROOT/formal_v3_resume_attempts
  PORT=29796
  CLI_BOUNDARY=()
fi

case "$EXPECTED_FINAL_HEALTH_SHA256" in
  (*[!0-9a-f]*|'') echo "FATAL: final-health hash is not pinned"; exit 44 ;;
esac
[ "${#EXPECTED_FINAL_HEALTH_SHA256}" -eq 64 ]
for REQUIRED in \
  "$FINAL_HEALTH" "$STATIC" "$STATIC_CONTRACT" "$PLAN_AUDIT" \
  "$OVERLAY_VERIFY" "$FREEZE" "$PLAN/header.json" "$PLAN/checksums.json" \
  "$CFG" "$V1_YAML" "$V2_YAML" "$YAML_SELF" "$ADDENDUM" "$RUNNER" \
  "$BOOT_RECEIPT" "$RECEIPT" "$GPIC_PREFLIGHT"; do
  test -f "$REQUIRED" && test ! -L "$REQUIRED" \
    || { echo "FATAL: formal-v3 prerequisite missing/non-regular: $REQUIRED"; exit 43; }
done
if grep -Eq '^[[:space:]]+init_from:' "$CFG"; then
  echo "FATAL: trainer.init_from is forbidden"
  exit 42
fi
LIVE_HEALTH_SHA=$(sha256sum "$FINAL_HEALTH" | awk '{print $1}')
[ "$LIVE_HEALTH_SHA" = "$EXPECTED_FINAL_HEALTH_SHA256" ] \
  || { echo "FATAL: FINAL_HEALTH SHA differs from pinned authority"; exit 44; }
RECEIPT_SHA=$(sha256sum "$RECEIPT" | awk '{print $1}')
FREEZE_SHA=$(sha256sum "$FREEZE" | awk '{print $1}')
PLAN_HEADER_SHA=$(sha256sum "$PLAN/header.json" | awk '{print $1}')

"$PY" - \
  "$MODE" "$RECEIPT" "$CK/last.pt" "$CK/metrics.jsonl" "$YAML_SELF" \
  "$ADDENDUM" "$RUNNER" "$FINAL_HEALTH" "$STATIC" "$STATIC_CONTRACT" \
  "$V1_YAML" "$CFG" "$PLAN" "$FREEZE" "$PLAN_AUDIT" "$OVERLAY_VERIFY" \
  "$EXPECTED_ARCH_FP_SHA256" "$MANIFEST_SHA256" "$EXACT_EXPOSURES" \
  "$RECEIPT_SCHEMA" "$EXPECTED_CHECKPOINT_STEP" "$EXPECTED_METRICS_STEP" <<'PYEOF'
import hashlib,json,pathlib,stat,sys,yaml
(
 mode,receipt_p,checkpoint,metrics,yaml_p,addendum,runner,health_p,static_p,
 contract_p,v1_yaml,config_p,plan,freeze_p,audit_p,overlay_p,arch,gpic,exact,
 schema,expected_checkpoint_step,expected_metrics_step,
)=sys.argv[1:]
paths=list(map(pathlib.Path,(receipt_p,checkpoint,metrics,yaml_p,addendum,runner,
 health_p,static_p,contract_p,v1_yaml,config_p,plan,freeze_p,audit_p,overlay_p)))
(
 receipt_p,checkpoint,metrics,yaml_p,addendum,runner,health_p,static_p,
 contract_p,v1_yaml,config_p,plan,freeze_p,audit_p,overlay_p,
)=paths
expected_checkpoint_step=int(expected_checkpoint_step)
expected_metrics_step=int(expected_metrics_step)
exact=int(exact)
load=lambda p: json.load(open(p,encoding="utf-8"))
receipt,health,static,contract,audit,overlay=map(load,(receipt_p,health_p,static_p,contract_p,audit_p,overlay_p))
header=load(plan/"header.json")
assert receipt.get("status")==health.get("status")==static.get("status")==contract.get("status")==audit.get("status")==overlay.get("status")=="PASS"
assert receipt.get("schema")==schema
assert int(receipt.get("expected_checkpoint_step",-1))==expected_checkpoint_step
assert int(receipt.get("exact_target_step",-1))==166667
assert int(receipt.get("exact_target_exposures",-1))==exact
assert int(receipt.get("world_size",-1))==64 and int(receipt.get("global_batch",-1))==2304
assert int(receipt.get("tag_only_exposure",-1))==0
assert int(receipt["metrics"]["last_step"])==expected_metrics_step
assert pathlib.Path(receipt["checkpoint"]["path"]).resolve()==checkpoint.resolve()
assert pathlib.Path(receipt["metrics"]["path"]).resolve()==metrics.resolve()
yaml_key="recover_yaml" if mode=="recover384" else "main_yaml"
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert receipt[yaml_key]["sha256"]==sha(yaml_p)
assert receipt["addendum_sha256"]==sha(addendum)
assert receipt["runner"]["sha256"]==sha(runner)
def identity(path):
 value=path.lstat(); assert stat.S_ISREG(value.st_mode) and not path.is_symlink()
 return {"device":value.st_dev,"inode":value.st_ino,"bytes":value.st_size,"mtime_ns":value.st_mtime_ns,"ctime_ns":value.st_ctime_ns}
assert identity(checkpoint)==receipt["checkpoint"]["identity"]
assert identity(metrics)==receipt["metrics"]["identity"]
assert not checkpoint.with_name(checkpoint.name+".tmp").exists()
assert health["architecture_fingerprint_sha256"]==arch
assert health["gpic_manifest_sha256"]==gpic
assert health["freeze_sha256"]==sha(freeze_p)
assert health["formal_plan_header_sha256"]==sha(plan/"header.json")
assert health["formal_plan_checksums_sha256"]==sha(plan/"checksums.json")
assert int(health["tag_only_exposure"])==0
assert static["formal_job"]["sha256"]==sha(v1_yaml)
assert static["config"]["config_sha256"]==sha(config_p)
assert int(audit["formal_prefix"]["steps"])==166667
assert int(audit["formal_prefix"]["exposures"])==exact
assert int(audit["full_stage"]["anime_tag_only_exposures"])==0
assert int(audit["multi_ar"]["active_native_bucket_count"])==20
assert int(overlay["checks"]["anime_full_sidecar_strict_nl"]["tag_only_selected"])==0
cfg=yaml.safe_load(config_p.read_text(encoding="utf-8"))
assert "init_from" not in cfg["trainer"] and cfg["trainer"]["resume"] is True
assert int(cfg["trainer"]["max_exposure_images"])==exact
assert int(cfg["trainer"]["batch_size"])==36
assert cfg["dataset"]["offline_plan_root"]==str(plan)
if mode=="recover384":
 replay=checkpoint.parent/"auto_resume_reconcile_v1"
 assert receipt["replay_guard_precondition"]["state"]=="ABSENT"
 assert not replay.exists() and not replay.is_symlink()
print("FORMAL_V3_CONTROL_PASS",mode,receipt["checkpoint"]["sha256"],receipt["metrics"]["sha256"])
PYEOF

HOST=$(hostname)
NODE_RANK=${HOST##*-}
JOBTAG=${HOST%-*}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
[ "$NODE_RANK" -lt "$NNODES" ] || { echo "FATAL: node rank out of range"; exit 21; }
ATTEMPT=$ATTEMPT_PARENT/$JOBTAG
mkdir -p "$ATTEMPT"
FRESH=$ATTEMPT/n$NODE_RANK.fresh
test ! -e "$FRESH" && test ! -L "$FRESH" \
  || { echo "FATAL: attempt marker collision"; exit 45; }
printf 'READY|%s|%s|%s|%s\n' "$JOBTAG" "$NODE_RANK" "$(date +%s)" "$HOST" > "$FRESH.tmp"
mv "$FRESH.tmp" "$FRESH"
NAMESPACE_PASS=$ATTEMPT/namespace.pass
if [ "$NODE_RANK" = 0 ]; then
  READY=0
  for _ in $(seq 1 900); do
    COUNT=0
    for RANK in $(seq 0 $((NNODES - 1))); do
      [ -f "$ATTEMPT/n$RANK.fresh" ] && [ ! -L "$ATTEMPT/n$RANK.fresh" ] && COUNT=$((COUNT + 1))
    done
    if [ "$COUNT" -eq "$NNODES" ]; then READY=1; break; fi
    sleep 2
  done
  [ "$READY" -eq 1 ] || { echo "FATAL: all-node fresh marker barrier timeout"; exit 45; }
  test -d "$CK" && test ! -L "$CK" && test -f "$CK/last.pt" && test ! -L "$CK/last.pt" \
    || { echo "FATAL: validated checkpoint namespace missing"; exit 42; }
  test -d "$RES" && test ! -L "$RES" \
    || { echo "FATAL: result namespace missing"; exit 42; }
  printf 'PASS|%s|%s|%s|%s\n' "$JOBTAG" "$LIVE_HEALTH_SHA" "$RECEIPT_SHA" "$(date +%s)" > "$NAMESPACE_PASS.tmp"
  mv "$NAMESPACE_PASS.tmp" "$NAMESPACE_PASS"
fi
for _ in $(seq 1 900); do
  [ -f "$NAMESPACE_PASS" ] && [ ! -L "$NAMESPACE_PASS" ] && break
  sleep 2
done
test -f "$NAMESPACE_PASS" && test ! -L "$NAMESPACE_PASS" \
  || { echo "FATAL: namespace authorization missing"; exit 45; }
IFS='|' read -r PASS_STATUS PASS_JOBTAG PASS_HEALTH PASS_RECEIPT PASS_TS < "$NAMESPACE_PASS"
[ "$PASS_STATUS" = PASS ] && [ "$PASS_JOBTAG" = "$JOBTAG" ] \
  && [ "$PASS_HEALTH" = "$LIVE_HEALTH_SHA" ] && [ "$PASS_RECEIPT" = "$RECEIPT_SHA" ] \
  || { echo "FATAL: namespace authorization mismatch"; exit 45; }

export OFFLINE_ARB_AUTO_RESUME_RECONCILE=1
export OFFLINE_ARB_ATTEMPT_JOBTAG="$JOBTAG"
export OFFLINE_ARB_ATTEMPT_NONCE="$RECEIPT_SHA"
export OFFLINE_ARB_AUTO_RESUME_FREEZE_SHA256="$FREEZE_SHA"
export OFFLINE_ARB_AUTO_RESUME_PLAN_HEADER_SHA256="$PLAN_HEADER_SHA"

GPIC_JSON=$ATTEMPT/gpicpreflight.n$NODE_RANK.$STAMP.json
"$PY" "$GPIC_PREFLIGHT" \
  --artifact-root "$ARTIFACT_ROOT" --buildtag "$ARTIFACT_BUILDTAG" \
  --manifest-sha256 "$MANIFEST_SHA256" --catalog-root "$CAT" \
  --plan-root "$PLAN" --config "$CFG" --job-yaml "$YAML_SELF" \
  --jobtag "$JOBTAG" --node-rank "$NODE_RANK" --output "$GPIC_JSON"

RDZV=$ATTEMPT/rdzv.addr
if [ "$NODE_RANK" = 0 ]; then
  MYIP=$(hostname -i | awk '{print $1}')
  printf '%s|%s|%s|%s|%s\n' "$MYIP" "$(date +%s)" "$JOBTAG" "$LIVE_HEALTH_SHA" "$RECEIPT_SHA" > "$RDZV.tmp"
  mv "$RDZV.tmp" "$RDZV"
  MASTER_ADDR=$MYIP
else
  MASTER_ADDR=""
  for _ in $(seq 1 900); do
    if [ -s "$RDZV" ] && [ ! -L "$RDZV" ]; then
      IFS='|' read -r CANDIDATE TS TOKEN HEALTH AUTH < "$RDZV"
      NOW=$(date +%s)
      if [ "$TOKEN" = "$JOBTAG" ] && [ "$HEALTH" = "$LIVE_HEALTH_SHA" ] \
        && [ "$AUTH" = "$RECEIPT_SHA" ] && [ "$TS" -le $((NOW + 60)) ] \
        && [ $((NOW - TS)) -lt 3600 ]; then
        MASTER_ADDR=$CANDIDATE
        break
      fi
    fi
    sleep 2
  done
  [ -n "$MASTER_ADDR" ] || { echo "FATAL: formal rendezvous timeout"; exit 20; }
fi

{
  echo "MODE=$MODE RUN_ID=$RUN_ID node=$NODE_RANK world=64 global_batch=2304 exact_exposures=$EXACT_EXPOSURES receipt_sha=$RECEIPT_SHA"
  hostname
  nvidia-smi -L
  df -h /local | tail -1
} 2>&1 | tee "$RES/probe.$JOBTAG.n$NODE_RANK.$STAMP.log"

(while true; do
   use=$(df --output=pcent /local 2>/dev/null | tail -1 | tr -dc '0-9' || true)
   if [ "${use:-0}" -ge 85 ]; then
     ts=$(date +%s)
     mv /local/evan/anime_img_cache /local/evan/anime_img_cache.old.$ts 2>/dev/null || true
     mkdir -p /local/evan/anime_img_cache
     nice -n 19 rm -rf /local/evan/anime_img_cache.old.$ts &
     echo "$(date -u) rotated node-local anime image cache at ${use}%"
   else
     echo "$(date -u) /local ${use}% ok"
   fi
   sleep 3600
 done) > "$RES/cachejanitor.$JOBTAG.n$NODE_RANK.$STAMP.log" 2>&1 &
(while true; do
   date -u
   [ -r /sys/fs/cgroup/memory.current ] && echo "mem.current: $(cat /sys/fs/cgroup/memory.current)"
   [ -r /sys/fs/cgroup/memory.pressure ] && cat /sys/fs/cgroup/memory.pressure
   df --output=avail /local | tail -1
   sleep 120
 done) > "$RES/nodemon.$JOBTAG.n$NODE_RANK.$STAMP.log" 2>&1 &

set +e
"$PY" -m torch.distributed.run \
  --nnodes "$NNODES" --nproc-per-node 8 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$PORT" \
  train.py --config "$CFG" "${CLI_BOUNDARY[@]}" \
  trainer.checkpoint_dir="$CK" \
  dataset.gpic_payload_backend=offline_gpic_blob_v1 \
  dataset.offline_gpic_blob_root="$ARTIFACT_ROOT" \
  dataset.gpic_blob_manifest_sha256="$MANIFEST_SHA256" \
  dataset.gpic_blob_fd_cache_size=1024 \
  2>&1 | tee "$RES/train.$JOBTAG.n$NODE_RANK.$STAMP.log"
PIPE_RC=("${PIPESTATUS[@]}")
set -e
[ "${PIPE_RC[1]}" -eq 0 ] || { echo "FATAL: formal evidence tee failed"; exit "${PIPE_RC[1]}"; }
exit "${PIPE_RC[0]}"
