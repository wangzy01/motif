# DPG-Bench evaluation kit (MoT stage-256/512)

Self-contained DPG-Bench prompt-following eval for the 4 ablation runs
(`v1`, `b1`, `b2`, `b3`). Generate images on the 1065 DPG prompts, then judge
prompt adherence with a VQA model over the DPG yes/no question graph.

## TL;DR — which scorer, and is it comparable?

**Scorer = Qwen3-VL-2B-Instruct** (local `/fsx/models/Qwen3-VL-2B-Instruct`)
used as a VQA judge.

> **Comparability verdict: INTERNALLY-CONSISTENT ONLY.**
> These scores are valid for *relative* comparison between our runs
> (B1-vs-v1, B2, B3) and for our own positioning. They are **NOT directly
> comparable** to published DPG-Bench leaderboard numbers, which use the
> original **mPLUG** VQA model.

### Why not the original mPLUG scorer?

The official DPG scorer (`compute_dpg_bench.py`) is mPLUG VQA via `modelscope`,
which depends on **`fairseq`** + `taming-transformers-rom1504` + `librosa` +
`decord`. `fairseq` has **no Python 3.12 wheel** and is effectively unmaintained
for 3.12. This box has **only `python3.12`** (`ls /usr/bin/python3.*` → just
`python3.12`; no `conda`, no `python3.10`). Building the legacy stack would mean
sourcing a 3.10 interpreter we don't have. It is **infeasible here**, so we
pivoted to a modern local VLM judge that answers the *same* DPG questions.

The scoring **protocol** (below) is faithful to the upstream
`compute_dpg_one_sample`; only the VQA backend differs.

## Scoring protocol (faithful to `compute_dpg_bench.py`)

For each generated image (keyed to its DPG `item_id`):
1. Ask the judge every DPG question for that prompt. Answer → `1.0` if "yes"
   else `0.0` (`parse_yes_no`, robust to extra words).
2. **Dependency zeroing**: a question's score is forced to `0` if ANY of its
   parent questions (its `dependency` list, ignoring parent id `0`) was
   answered "no". (Mirrors the upstream parent-no → child-zero loop.)
3. Per-image score = mean over that prompt's questions (after zeroing).
4. **DPG overall** = mean over images × 100.
5. **Category breakdown** uses the *pre-zeroing* yes-rates (upstream
   `qid2scores_orig`): L1 = `tuple.split('-')[0]`
   (entity/attribute/relation/global/other), L2 = `tuple.split('(')[0]`.

Faithful detail: like upstream, `build_dpg_data.py` **drops the first CSV data
row** (`if i==0: continue`). Result: 1065 items, 14391 questions.

Intentional difference: upstream tiles a **2×2 grid** (`pic_num=4`) and crops
quadrants. We generate **one image per prompt** (the whole frame, `pic_num=1`) —
the standard single-image setting and what `generate_dpg.py` produces.

## Files

| file | runs in | purpose |
|---|---|---|
| `dpg_bench.csv` | — | upstream prompt+QA graph (TencentQQGYLab/ELLA), copied for self-containment |
| `build_dpg_data.py` | any py (stdlib) | CSV → `dpg_prompts.jsonl` (clean, keyed by `item_id`) |
| `dpg_prompts.jsonl` | — | 1065 items, each `{id, prompt, questions:[{qid, dependency, tuple, l1, l2, question}]}` |
| `generate_dpg.py` | **venv-gpu** | sample our runs over DPG prompts (same EMA/snapshot/run-spec path as `sample_evalset.py`) |
| `score_dpg.py` | **venv-dpg** | Qwen3-VL judge + DPG aggregation (`score_dir` entry point) |
| `run_dpg.py` | **venv-dpg** | driver: score run dirs → `comparison.md`/`comparison.json` |

Outputs land in `experiments/results/dpg_bench/` (small): `comparison.md/json`,
`dpg_<run>.json`, `dpg_<run>_detail.jsonl`, and `samples/<run>/*.png`.

## The isolated scorer venv

**Path: `/fsx/evan/mot_code/venv-dpg`** (Python 3.12, ~5.0 GB).
Built SEPARATELY from the live training venv — **`venv-gpu` was never touched**.

Exact commands that worked:

```bash
python3.12 -m venv /fsx/evan/mot_code/venv-dpg
/fsx/evan/mot_code/venv-dpg/bin/python -m pip install --upgrade pip wheel setuptools
/fsx/evan/mot_code/venv-dpg/bin/pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  torch==2.11.0 torchvision==0.26.0 \
  "transformers==5.11.0" accelerate pillow numpy safetensors
```

Resulting stack (mirrors the proven training stack so Qwen3-VL loads identically):
`torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `transformers 5.11.0`,
`accelerate 1.14.0`, `pillow 12.2.0`, `numpy 2.5.0`, `safetensors 0.8.0`.
`pip check` → "No broken requirements found."

Smoke test (CPU; on a GPU node it is far faster):
```bash
# loads Qwen3-VL from the local /fsx path and answers one DPG-style question
/fsx/evan/mot_code/venv-dpg/bin/python - <<'PY'
import sys; sys.path.insert(0, "experiments/eval/dpg")
import score_dpg
from PIL import Image
j = score_dpg.QwenVLJudge(device="cpu")       # use device="cuda" on a GPU node
print(j.vqa(Image.new("RGB",(64,64),(220,20,20)), "Is the image red?"))  # -> 'yes'
PY
```
Verified output: judge loads in ~13 s, answers `'yes'` (parsed 1.0); full
`score_dir` on a dummy image runs the dependency/category aggregation end-to-end.

## How to run

Generation uses **venv-gpu** (proven training stack); scoring uses **venv-dpg**.
A GPU is required. The job `jobs/eval_dpg_bench.yaml` wires both steps; or run
the steps by hand on a GPU node:

### Relative 256 B1-vs-v1 pass (fast)
```bash
cd /fsx/evan/mot_code/mot_pilot_v1
export PYTHONPATH=$PWD
# 1) generate (venv-gpu). --limit caps prompts; drop --limit for full 1065.
/fsx/evan/mot_code/venv-gpu/bin/python experiments/eval/dpg/generate_dpg.py \
  --runs v1 b1 --config configs/stage1a_t2i_256.yaml --size 256 \
  --steps 64 --cfg 4.0 --weights ema --batch-size 8 --limit 24 \
  --out-dir experiments/results/dpg_bench/samples
# 2) score (venv-dpg)
/fsx/evan/mot_code/venv-dpg/bin/python experiments/eval/dpg/run_dpg.py \
  --samples-dir experiments/results/dpg_bench/samples --runs v1 b1 \
  --out-dir experiments/results/dpg_bench
```

### Full eval at 512 (all 4 runs, all 1065 prompts)
Only change the config + size for generation; scoring is resolution-agnostic:
```bash
cd /fsx/evan/mot_code/mot_pilot_v1
export PYTHONPATH=$PWD
# generate at 512 (the model is resolution-agnostic; the 512 config only changes
# resolution/shift/steps/batch). --limit 0 = all 1065 prompts.
/fsx/evan/mot_code/venv-gpu/bin/python experiments/eval/dpg/generate_dpg.py \
  --runs v1 b1 b2 b3 --config configs/stage1a_t2i_512.yaml --size 512 \
  --steps 64 --cfg 4.0 --weights ema --batch-size 8 --limit 0 \
  --out-dir experiments/results/dpg_bench_512/samples
# score
/fsx/evan/mot_code/venv-dpg/bin/python experiments/eval/dpg/run_dpg.py \
  --samples-dir experiments/results/dpg_bench_512/samples --runs v1 b1 b2 b3 \
  --out-dir experiments/results/dpg_bench_512
```

On the cluster, prefer the job (it snapshots ckpts to `/local` first to avoid the
training atomic-replace race): `jcc job submit jobs/eval_dpg_bench.yaml`
(default `LIMIT=24`, `SIZE=256`; edit `LIMIT=0` / `SIZE=512` + `CONFIG` in the
YAML to scale up).

## Safety notes

Read-only w.r.t. training: never writes checkpoints/configs/networks; loads EMA
weights with `strict=False`; on the cluster, generation reads `/local` snapshots
of the live ckpts. `venv-gpu` is never modified.

---

# mPLUG comparability path (ORIGINAL DPG scorer) — leaderboard-comparable

> **UPDATE (added 2026-06-22):** the legacy mPLUG/fairseq/modelscope stack DID
> install, in a **separate conda env** (not the py3.12 box pythons). So we now
> *also* offer the **original mPLUG VQA scorer**, whose numbers are directly
> comparable to published DPG-Bench leaderboards. The Qwen3-VL path above is
> unchanged and remains the default (fast, internally-consistent ablation).

## Comparability verdict (now)

> **LEADERBOARD-COMPARABLE — iff you use BOTH:**
> 1. the **mPLUG scorer** (`score_dpg_mplug.py` / `run_dpg.py --backend mplug`), AND
> 2. the **2×2-grid `pic_num=4`** image layout (`generate_dpg.py --grid 2`),
>    which is exactly what the upstream `compute_dpg_bench.py` expects.
>
> Using mPLUG with single images (`--grid 1` / `--pic-num 1`) is still the
> original mPLUG model, but `pic_num=1` rather than the leaderboard's `pic_num=4`.
> The Qwen3-VL default remains **internally-consistent only**.

## The separate conda env (do NOT confuse with venv-gpu / venv-dpg)

**Path: `/fsx/evan/miniconda3/envs/dpg-mplug`** (Python **3.10**, ~4.8 GB).
Built as a fresh miniconda + conda-forge env; **`venv-gpu` and `venv-dpg` were
never touched**. The legacy stack needs Python 3.10 (no `fairseq` wheel for
3.12), which is why it lives in conda, not in either venv.

Resulting stack (the versions that actually work together):
`torch 2.0.1+cu118`, `torchvision 0.15.2+cu118`, `torchaudio 2.0.2+cu118`
(cu118 wheels are compiled for **sm_90**, so they run on H100),
`fairseq 0.12.2`, `modelscope 1.9.5`, `transformers 4.33.3`,
`tokenizers 0.13.3`, `huggingface-hub 0.16.4`, `numpy 1.26.4`,
`pyarrow 12.0.1`, `datasets 2.13.0`, `timm 0.9.12`, `ftfy 6.1.3`,
`taming-transformers-rom1504 0.0.6`, `opencv-python-headless 4.9.0.80`,
`omegaconf 2.0.6`, `hydra-core 1.0.7`, `librosa 0.10.1`, `decord 0.6.0`,
`soundfile 0.14.0`. `pip check` is clean except one benign
note: `taming-transformers-rom1504 requires pytorch-lightning` — we deliberately
omit pytorch-lightning because (a) it is NOT needed for the VQA inference path
and (b) installing it re-clobbers torch to a newer CPU build.

### Exact build commands that worked

```bash
# 1) miniconda + a python 3.10 env (conda-forge avoids the Anaconda-channel ToS prompt)
bash Miniconda3-latest-Linux-x86_64.sh -b -p /fsx/evan/miniconda3
/fsx/evan/miniconda3/bin/conda create -y -n dpg-mplug -c conda-forge --override-channels python=3.10
PY=/fsx/evan/miniconda3/envs/dpg-mplug/bin/python
$PY -m ensurepip --upgrade           # conda-forge py3.10 ships no pip by default
$PY -m pip install wheel ninja        # wheel (bdist_wheel) + ninja for fairseq's C++ build

# 2) torch FIRST (cu118 -> sm_90/H100 capable), build deps for fairseq
$PY -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118
$PY -m pip install "cython<3" "setuptools==59.5.0"

# 3) fairseq 0.12.2 — the hard one. The PyPI sdist is BROKEN (its setup.py
#    references clib/*.cpp files that are NOT shipped in the tarball ->
#    "balanced_assignment.cpp: No such file or directory"). Build from the
#    GitHub *source tree* (which includes clib/), with --no-build-isolation so
#    setup.py sees torch + cython.
git clone --depth 1 --branch v0.12.2 https://github.com/facebookresearch/fairseq.git
$PY -m pip install --no-build-isolation ./fairseq            # compiles the Cython + C++ exts
$PY -m pip install --index-url https://download.pytorch.org/whl/cu118 torchaudio==2.0.2+cu118
                                                             # (fairseq dragged in a mismatched torchaudio; pin it back)
$PY -m pip install "numpy==1.26.4"                           # torch 2.0.1 + modelscope need numpy<2

# 4) modelscope + the mPLUG pipeline deps. taming/pytorch-lightning try to
#    clobber torch -> we restore the pinned torch afterward and downgrade the
#    sub-deps that demand numpy>=2 / pyarrow>=14.
$PY -m pip install "modelscope==1.9.5"
$PY -m pip install "transformers==4.33.3" "tokenizers==0.13.3" "huggingface-hub==0.16.4"
$PY -m pip install "timm==0.9.12" "ftfy==6.1.3" "taming-transformers-rom1504==0.0.6" \
    "rapidfuzz" "zhconv" "tiktoken" "transformers_stream_generator" "unicodedata2"
$PY -m pip install --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 numpy==1.26.4
$PY -m pip install "opencv-python-headless==4.9.0.80"        # 4.13 demands numpy>=2
$PY -m pip install "pyarrow==12.0.1"                         # datasets 2.13 uses pa.PyExtensionType (gone in pa>=14)

# 5) the modelscope multi_modal pipeline EAGERLY imports the OFA model + the
#    shared multimodal preprocessor, which need librosa + decord even for the
#    image-only VQA task. Install them (both are in the upstream requirements).
$PY -m pip install "librosa==0.10.1" soundfile "decord>=0.6.0"
```

### Verified smoke test (CPU, on this login node)

```text
PIPELINE CONSTRUCTED OK (model files loaded)   # <All keys matched successfully>
VQA answer: 'yes'                              # vqa(red_image, "Is the image red?")
score_one_image per_image_score: 1.0  crop_scores: [1.0, 1.0, 1.0, 1.0]  # pic_num=4 path
SMOKE TEST OK
```
The pipeline loads the 2.8 GB weight from the /fsx cache and answers correctly on
CPU; GPU inference is identical (`device="gpu"`). The full DPG run needs a GPU
(speed), but construction + inference are proven here end-to-end. NOTE: pass the
model's LOCAL cache directory (score_dpg_mplug.py resolves `<cache>/<model_id>`
automatically) — passing the bare model id makes modelscope 1.9.5 re-run its
download/integrity machinery, which can re-fetch the 2.8 GB weight and stall.

One small site-packages patch was needed in THIS env (not in the project):
modelscope 1.9.5's `hub/file_download.py` rejects files served with
`Content-Length: None` (chunked; modelscope.cn does this for some small text
files) even though they download fully. The strict check was relaxed to tolerate
`total in (None, 0)`. This only affects downloading; the cluster job reads the
already-cached model and never hits it.

## mPLUG model cache

**Path: `MODELSCOPE_CACHE=/fsx/evan/modelscope_cache`** (set this env var so the
cluster node reads the model offline). The model
`damo/mplug_visual-question-answering_coco_large_en` (the original DPG VQA model;
on modelscope.cn, **not** on HF) lives at
`/fsx/evan/modelscope_cache/damo/mplug_visual-question-answering_coco_large_en/`
— dominated by `pytorch_model.bin` (~2.78 GB).

## Footprint added (/fsx) — measured

* `/fsx/evan/miniconda3` (base + `dpg-mplug` env + retained pkgs): **7.5 GB**
  (the env alone is ~6.3 GB; the pip build cache was deleted after install).
* `/fsx/evan/modelscope_cache` (mPLUG model, `pytorch_model.bin` ~2.8 GB): **2.8 GB**.
* **Total new /fsx footprint ≈ 10.3 GB** (well under the ~50 GB budget).

## How to run the mPLUG comparability path

Generation uses **venv-gpu** with `--grid 2`; scoring uses the **conda
`dpg-mplug`** python with `--backend mplug --pic-num 4`. A GPU is required for
mPLUG inference. The job `jobs/eval_dpg_bench_mplug.yaml` wires both
(do NOT confuse with the Qwen3-VL `jobs/eval_dpg_bench.yaml`).

### FULL leaderboard-comparable pass at 512 (all 4 runs, all 1065 prompts)
```bash
cd /fsx/evan/mot_code/mot_pilot_v1
export PYTHONPATH=$PWD
export MODELSCOPE_CACHE=/fsx/evan/modelscope_cache
# 1) GENERATE the 2x2 grid at 512 (per-tile = 512 -> saved PNG is 1024x1024).
/fsx/evan/mot_code/venv-gpu/bin/python experiments/eval/dpg/generate_dpg.py \
  --runs v1 b1 b2 b3 --config configs/stage1a_t2i_512.yaml --size 512 \
  --steps 64 --cfg 4.0 --weights ema --batch-size 8 --limit 0 --grid 2 \
  --out-dir experiments/results/dpg_bench_mplug_512/samples
# 2) SCORE with the ORIGINAL mPLUG (conda env), pic_num=4. --resolution is
#    inferred as width//2 (=512) for the 2x2 grid, so no flag needed.
/fsx/evan/miniconda3/envs/dpg-mplug/bin/python experiments/eval/dpg/run_dpg.py \
  --backend mplug --pic-num 4 --device gpu \
  --samples-dir experiments/results/dpg_bench_mplug_512/samples --runs v1 b1 b2 b3 \
  --out-dir experiments/results/dpg_bench_mplug_512
```

### Relative 256 B1-vs-v1 pass (fast, still mPLUG + grid)
```bash
cd /fsx/evan/mot_code/mot_pilot_v1
export PYTHONPATH=$PWD
export MODELSCOPE_CACHE=/fsx/evan/modelscope_cache
# 1) generate (venv-gpu) — 2x2 grid at 256; --limit caps prompts (drop for full 1065).
/fsx/evan/mot_code/venv-gpu/bin/python experiments/eval/dpg/generate_dpg.py \
  --runs v1 b1 --config configs/stage1a_t2i_256.yaml --size 256 \
  --steps 64 --cfg 4.0 --weights ema --batch-size 8 --limit 24 --grid 2 \
  --out-dir experiments/results/dpg_bench_mplug/samples
# 2) score (conda dpg-mplug, mPLUG backend)
/fsx/evan/miniconda3/envs/dpg-mplug/bin/python experiments/eval/dpg/run_dpg.py \
  --backend mplug --pic-num 4 --device gpu \
  --samples-dir experiments/results/dpg_bench_mplug/samples --runs v1 b1 \
  --out-dir experiments/results/dpg_bench_mplug
```

On the cluster: `jcc job submit jobs/eval_dpg_bench_mplug.yaml` (default
`LIMIT=24`, `SIZE=256`, `GRID=2`, `PICNUM=4`; set `LIMIT=0` / `SIZE=512` +
`CONFIG` to scale to the full leaderboard pass).

## mPLUG files

| file | runs in | purpose |
|---|---|---|
| `score_dpg_mplug.py` | **conda `dpg-mplug`** | ORIGINAL mPLUG VQA scorer + upstream 2×2-crop / dependency-zeroing / L1-L2 aggregation (`score_dir` entry point, same shape as `score_dpg.py`) |
| `generate_dpg.py --grid 2` | **venv-gpu** | additive 2×2-grid (`pic_num=4`) image layout the original expects (default `--grid 1` unchanged) |
| `run_dpg.py --backend mplug` | **conda `dpg-mplug`** | additive backend switch to the mPLUG scorer (default `--backend qwen3vl` unchanged) |
| `jobs/eval_dpg_bench_mplug.yaml` | jcc | generate (grid, venv-gpu) → score (mPLUG, conda) wiring |
