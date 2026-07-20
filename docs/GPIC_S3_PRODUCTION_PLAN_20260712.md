# GPIC full S3 blob production plan — 2026-07-12

## Frozen user decisions

- Project FSx usage is a hard maximum of 800 GB; current usage is about 500 GB.
- Store the full GPIC payload as large immutable objects under `/s3/evan`.
- Use an explicit row-id fixed-width index.
- Use the fast freeze policy: one full fresh verification before publication;
  formal preflight revalidates the compact controls and fingerprints, while
  every runtime segment is SHA-256 checked.
- Formal target remains 8 nodes × 8 GPUs, global batch 2304, 166,667 steps,
  with 2.50–2.55 s/step probation target and 2.592 s/step five-day line.

## Measured facts

- GPIC catalog rows: 20,000,000.
- The 100k cold pilot produced 14,502,888,558 payload bytes and sustained
  125.605 images/s on one CPU16 worker.
- Full payload estimate from that sample: about 2.90 TB decimal.
- Eight-node `/local/evan` topology job:
  `evan-mot-crop-offlinearb-local-topology-94ns`.
- Topology summary raw SHA-256:
  `27c48e9d6438d992fa850a0da9192c7043ccb6714e65ffa33460a755b4c5d73a`.
- `/local/evan` is a separate ext4 mount, not the 500 GB container `/local`.
  Each of eight distinct nodes had 30,277,923,348,480 total bytes and
  18.18–24.05 TB available.  The conservative minimum safe staging budget was
  12,128,305,401,856 bytes per worker.
- Bounded Mountpoint-S3 build and fresh replay passed byte/SHA closure.  The
  bounded S3 result is diagnostic only; production still requires the scale,
  full verifier, W64, and probation gates below.

## Storage budgets

| Plane | Hard budget | Intended use |
|---|---:|---|
| FSx incremental | 50 GB | dense index, fragments, manifests, receipts, telemetry, verifier report |
| S3 payload | 3.2 TB | full GPIC blobs plus compact controls |
| `/local/evan` per builder worker | 512 GB | one complete 1/8 build partition before upload |
| Dense index | exactly 1,120,000,000 bytes | 20M × 56-byte explicit records |

The 50 GB FSx sub-budget leaves substantial margin below the user's 800 GB
project cap.  No GPIC payload blob may be published to FSx.

## Production record and key layout

The dense index record is little-endian:

```text
<global_row_id:i64, blob_id:u32, offset:u64, length:u32, segment_sha256:32B>
```

Record size is exactly 56 bytes.  The record at dense slot `i` must contain
`global_row_id == gpic_row_base + i`; this detects row/index rebinding directly.

Workers own catalog-local rows by `local_row_id % 8 == worker_id`.  Each worker
sorts its rows by canonical source path for source-read efficiency.  Global
blob IDs are deterministic and collision-free:
`blob_id = worker_id + 8 * worker_blob_sequence`.  They are strictly ascending
and unique in the canonical manifest but may be sparse at the tail when workers
finish with different blob counts.  The canonical relative name is
`workers/worker-{blob_id % 8}/blob-{blob_id:08d}.bin`.

All S3 keys live under one high-entropy, never-reused build namespace:

```text
/s3/evan/artifacts/offline_gpic_blob_v1/<build_jobtag>/
  _INIT.json
  _SUCCESS
  workers/worker-0/blob-00000000.bin
  ...
  controls/manifest.json
```

The committed S3 root exact set is `_INIT.json`, `_SUCCESS`, `workers/`, and
`controls/`; the `controls/` exact set is only `manifest.json`.  The FSx control root
separately contains the byte-identical canonical manifest, its own `_SUCCESS`,
the dense index, and verifier report.  `release.storage.s3_init` binds the exact
raw `_INIT.json` byte count and SHA-256.

No two workers write the same key.  Only the coordinator writes root controls.
The build must not rely on Mountpoint coordinating concurrent writes to one key.

## Build protocol

### Phase A — source to node-local partition

1. Bind live runner/job YAML, catalog header/hash, GPIC parquet SHA/schema,
   topology report jobtag/raw SHA, `/local/evan` ext4 mount identity, and all
   storage limits before reading payloads.
2. Materialize exactly the worker's 2.5M row identities without loading image
   payloads into aggregate memory; path-sort the partition.
3. Read with the scale-probe-selected thread count.  Empty/short/changed reads
   fail; no skip, replacement, retry to a different backend, or row fallback.
4. Write 4 GiB target blobs and a worker index fragment under a unique
   `/local/evan` work root.  Hash every segment and whole blob while writing.
5. Fsync every closed blob, the fragment, manifest, and local directory.
6. Validate the worker row congruence, exact count, blob closure, total bytes,
   hard budgets, and immutable local fingerprints.

The complete approximately 360–370 GB worker partition remains local until its
S3 upload is closed.  This decouples small-file reads from Mountpoint uploads
and permits bounded same-pod retransmission without rereading the source.

### Phase B — worker-exclusive S3 upload

1. Recheck local fingerprints and SHA before each upload attempt.
2. Require the destination key to be fresh; create and write sequentially.
3. Fsync the S3 file to complete multipart upload, then verify visible size.
4. Retries, if any, stay in the same pod and may only target the same intended
   immutable key after positively proving that no completed key exists.
5. A visible key after an error is an unknown/committed state, never an ordinary
   failure eligible for overwrite.
6. Publish a compact worker receipt to FSx.  Do not publish root `_SUCCESS`.

Local blobs may be removed only after the S3 worker receipt is durable.  Their
cross-job persistence is never assumed.

### Phase C — coordinator closure

1. Wait for eight exact worker receipts; bind jobtag, worker IDs, code/YAML,
   boot identities, topology evidence, counts, bytes, and S3 key set.
2. Prove 20M-row union closure and no overlap.  Enforce byte imbalance and
   global 3.2 TB / FSx 50 GB budgets.
3. Merge fragments into the 1,120,000,000-byte dense index.  At slot `i`, read
   the next record from worker `i % 8` and require the explicit row ID.
4. Write/fsync the dense index and pre-publication manifest on FSx.  Publish a
   build-complete receipt, but still no production `_SUCCESS`.

## Full fresh verifier and publication

A separate 8×CPU16 job on fresh physical hosts is the single full verifier
authorized by the fast-freeze decision.

1. Partition whole S3 blobs across verifier workers.
2. Read every blob sequentially from S3 and recompute its whole SHA-256.
3. Use worker blob-order sidecars to recheck every segment boundary, explicit
   row ID, length, and segment SHA against the dense index.
4. Close all blob bytes, all 20M records, exact key set, and verifier telemetry.
5. Coordinator publishes the byte-identical canonical manifest to S3 and FSx,
   then writes each plane's `_SUCCESS` as that plane's final entry.  Both marker
   payloads are exactly the raw manifest-file SHA-256 plus one newline.  A
   post-marker error is committed/unconfirmed and may never create contradictory
   failure evidence.

The production artifact is nonexistent to readers until both FSx and S3
`_SUCCESS` files bind the same canonical manifest and verifier report.

## Runtime integration

- Only GPIC raw-byte transport changes.  Plan, catalog, caption sidecar,
  decode, native-size check, crop/resize, prompt/CFG RNG, and resume remain
  unchanged.
- Runtime lookup uses the planned global row ID, never the bucket-local decoder
  value.
- The dense index is read-only mmap; it must not become a Python list/dict.
- Startup hashes the already-open index FD that backs the mmap and rejects any
  live-path inode replacement before collectively authorizing reads.
- Per-process FD caching uses leases so eviction cannot close an FD during
  `pread`.  The FD cap is explicit and frozen.  W64/probation first tests a cap
  covering the complete manifest blob set (anticipated candidate: 1024), but
  startup must close that choice against `RLIMIT_NOFILE`, actual FD count, and
  RSS on every rank; a lower cap remains a measured fallback, not a silent one.
- Every `pread` is exact and every returned segment is SHA-256 checked.
- Selected blob failure is fatal.  There is no fallback to the original GPIC
  small file.
- Metrics include requested/pread bytes, blob opens/cache/evictions, per-window
  wall time, integrity/short-read counts, RSS/FD, and all-rank slowest supply.

## Remaining gates

1. Scale static/negative gate.
2. Eight-node ABBA source → `/local/evan` → S3 scale probe:
   efficiency hard gate 0.75, target 0.85; slow/fast node ratio at most 1.25.
3. Production builder/static tests and independent red-team review.
4. Full S3 build and fresh full verifier.
5. Source-versus-overlay semantic A/B, including fresh and resume hashes for
   trace, prompts, pixels, and CFG keep.
6. W64/64-process plan-shaped replay; slowest node capacity at least 1.25,
   zero integrity/fallback errors, bounded FD/RSS.
7. Independent 8×8 probation for 200–500 steps, checkpoint and exact resume.
8. Rebuild compact freeze controls without a second 2.9 TB rehash, submit
   formal, and monitor stable step time/data wait before declaring completion.

## Stop conditions

- FSx projected incremental usage above 50 GB for this artifact.
- S3 payload above 3.2 TB.
- Any missing/duplicate row, source identity drift, short read, SHA mismatch,
  non-fresh key, unknown marker state treated as ordinary failure, or fallback.
- Scale efficiency below 0.75, unacceptable node skew, or W64/probation speed
  incompatible with the five-day target.
