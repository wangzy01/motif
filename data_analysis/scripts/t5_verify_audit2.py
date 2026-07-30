"""Read-only verification of the 2nd external audit's concrete claims against the
ALREADY-PUBLISHED rich_v1 + synthetic_v1 (not the updated scripts). CPU-only.

Claims to check:
- published rich_v1 tok_* dtype is still int16
- MONET negative tok_* count (audit: 18 -> 17 gemini + 1 original)
- MONET rows with has_t4==1 but NO captioner with a positive T4 token (audit: 13)
- MONET aes null/NaN count (audit: 5,468)
- GPIC tok_i1 negatives (audit: none)
- synthetic_v1 holdout==1 count (audit: 6,565)
"""
import glob, json, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow.parquet as pq

S3 = os.environ.get('MOT_FILTER_OUT', '/s3-pixai-test-uw2/evan/mot_filtering_v1')
RICH = f'{S3}/t5_manifest/rich_v1'
SYN = f'{S3}/t5_manifest/synthetic_v1'
TOKS = ['tok_original', 'tok_florence', 'tok_internvl', 'tok_sharegpt4v', 'tok_gemini']
W = int(os.environ.get('READ_WORKERS', '48'))
t0 = time.time()


def main():
    mfiles = sorted(glob.glob(f'{RICH}/monet/*.parquet'))
    gfiles = sorted(glob.glob(f'{RICH}/gpic/*.parquet'))
    sfiles = sorted(glob.glob(f'{SYN}/monet_synth/*.parquet'))
    dtypes = {c: str(pq.read_schema(mfiles[0]).field(c).type) for c in TOKS + ['has_t4', 'aes']}
    gdtype = {c: str(pq.read_schema(gfiles[0]).field(c).type) for c in ['tok_i1']}

    def scan_m(f):
        d = pq.read_table(f, columns=TOKS + ['has_t4', 'aes']).to_pydict()
        toks = np.stack([np.asarray(d[c], dtype=np.int64) for c in TOKS])  # (5,N)
        neg_per = {c: int((np.asarray(d[c], dtype=np.int64) < 0).sum()) for c in TOKS}
        posT4 = (toks >= 129).sum(axis=0)            # captioners with a real (positive) T4 token
        h4 = np.asarray(d['has_t4'], dtype=np.int64)
        t4gap = int(((h4 == 1) & (posT4 == 0)).sum())
        aesnull = int(np.isnan(np.asarray(d['aes'], dtype=np.float64)).sum())
        return neg_per, t4gap, aesnull, len(h4)

    neg = {c: 0 for c in TOKS}; t4gap = 0; aesnull = 0; nrows = 0
    with ThreadPoolExecutor(max_workers=W) as ex:
        for npc, g, a, n in ex.map(scan_m, mfiles):
            for c in TOKS:
                neg[c] += npc[c]
            t4gap += g; aesnull += a; nrows += n
    print(f'[verify] MONET scanned {nrows:,} ({time.time()-t0:.0f}s)', flush=True)

    def scan_g(f):
        d = pq.read_table(f, columns=['tok_i1']).to_pydict()
        return int((np.asarray(d['tok_i1'], dtype=np.int64) < 0).sum())
    gneg = 0
    with ThreadPoolExecutor(max_workers=W) as ex:
        for r in ex.map(scan_g, gfiles):
            gneg += r

    def scan_s(f):
        d = pq.read_table(f, columns=['holdout']).to_pydict()
        return int((np.asarray(d['holdout'], dtype=np.int64) == 1).sum())
    shold = 0
    with ThreadPoolExecutor(max_workers=W) as ex:
        for r in ex.map(scan_s, sfiles):
            shold += r

    out = {
        'published_rich_monet_tok_dtypes': dtypes,
        'published_rich_gpic_tok_i1_dtype': gdtype,
        'monet_negative_tok_per_captioner': neg,
        'monet_negative_tok_total': sum(neg.values()),
        'monet_has_t4_but_no_positive_T4_captioner': t4gap,
        'monet_aes_null_or_nan': aesnull,
        'gpic_tok_i1_negative': gneg,
        'synthetic_holdout_eq_1': shold,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
