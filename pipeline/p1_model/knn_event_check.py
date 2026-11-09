"""Look-alike ambiguity counted per fast-wind STREAM instead of per hour (reviewer comment 5, 2026-10-17).

The hourly fast->slow share (PAPER_3 §6, Figure 5) counts every fast hour; hours inside one stream are strongly
autocorrelated, so long streams dominate. Here the fast query hours (ACE >= 550 km/s) are grouped into streams:
consecutive query rows with v >= 550 separated by less than 12 h belong to one stream. Per stream: the share of all its
look-alikes followed by slow wind (<= 400 km/s). Reported: number of streams, mean and median per-stream share, the
fraction of streams whose share exceeds 25 % and 50 %, and the hourly share for reference.

Same machinery as the published analysis (lib/knn_core.py): fold-consistent pool (look-alikes only from other
development batches, |ΔT| >= 13 d), K = 50, standardised inputs, every 2nd row as a query.
Input sets: B0's inputs (daily 193 Å set-2a features at t_src and 24/48/72 h before, PFSS f_s, back-mapped
recurrence), the six current-frame features only, and the random-hour control.

  python pipeline/p1_model/knn_event_check.py              (fixed 96-h lag; KNN_ALIGN_BETA=1 ballistic, 0.25 partial)
Output: data/knn_degeneracy/event_check_beta0.csv (paper; other alignments: suffix _beta<b>, ballistic none)
"""
import os

import numpy as np
import pandas as pd

# Paper alignment (decision 2026-10-23): look-alike source cell = central-meridian column 96 h before the ACE hour
# (KNN_ALIGN_BETA=0), as in the feature ranking; KNN_ALIGN_BETA=1 gives the earlier ballistic matching.
os.environ.setdefault('KNN_ALIGN_BETA', '0')
import knn_core as K  # noqa: E402

B = K.base_inputs()
sets = [('B0 information, 26 values (193 at 0/24/48/72 h + fs + bm)', np.concatenate([B['l193'], B['fs'], B['bm']], 1)),
        ('6 current-frame 193 features', B['c193'])]
keep = np.isfinite(sets[0][1]).all(1)                 # common rows for both sets
rng = np.random.default_rng(0)
rows = []
for label, X in sets + [('random hours (control)', None)]:
    if X is None:
        qi, _ = K.neighbours(sets[1][1], keep)
        pool = np.flatnonzero(keep)
        NB = rng.choice(pool, size=(len(qi), 50))
    else:
        qi, NB = K.neighbours(X, keep)
    vq = K.v[qi]
    fast = vq >= 550
    tq = K.tday[qi][fast]
    slow_nb = (K.v[NB][fast] <= 400)                  # [n_fast, 50]
    order = np.argsort(tq)
    tq, slow_nb = tq[order], slow_nb[order]
    sid = np.concatenate([[0], np.cumsum(np.diff(tq) > 0.5)])     # new stream after a gap of more than 12 h
    per = pd.Series(slow_nb.mean(1)).groupby(sid).mean()
    rows.append(dict(inputs=label, n_fast_hours=int(fast.sum()), n_streams=int(per.size),
                     hourly_share=float(slow_nb.mean()), stream_mean_share=float(per.mean()),
                     stream_median_share=float(per.median()), streams_share_gt25=float((per > 0.25).mean()),
                     streams_share_gt50=float((per > 0.50).mean())))
R = pd.DataFrame(rows)
R.to_csv(f'data/knn_degeneracy/event_check{K.KB.ALIGN_TAG}.csv', index=False)
pd.set_option('display.width', 200)
print(R.round(3).to_string(index=False))
