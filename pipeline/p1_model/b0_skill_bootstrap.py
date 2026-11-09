"""Paired 27-day block-bootstrap intervals for differences in threshold-free skill (second review, item 15, 2026-10-17).

Same matched hours and methods as b0_skill_metrics.py. For each pair of methods and each fast-wind threshold
(500, 600 km/s): the difference in ROC AUC and in best-threshold TSS (each model's own prediction threshold, 300-800 km/s
in 5 km/s steps, re-optimised in every resample), with 95 % percentile intervals from 1000 resamples of 27-day blocks
(the blocks of b0_matched_S.py). A resample is a set of block weights, so every metric is a weighted sum over hours.

  python pipeline/p1_model/b0_skill_bootstrap.py
Output: <B0_RUN>/skill_bootstrap.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from b0_skill_metrics import two_seed
from eval_matched_baselines import ace_hourly
from frozen_model import B0_RUN

NBOOT = 1000
THR = np.arange(300, 801, 5)


def best_tss(p, o_pos, w):
    """Best TSS over THR with hour weights w."""
    P = p[None, :] >= THR[:, None]                                   # [n_thr, n]
    pos, neg = w * o_pos, w * ~o_pos
    return np.max(P @ pos / pos.sum() - P @ neg / neg.sum())


if __name__ == '__main__':
    ace = ace_hourly()
    D = two_seed(B0_RUN, ace)
    M = two_seed(B0_RUN.replace('/B0', '/B0_mse'), ace)
    D = D.merge(M[['time', 'nn']].rename(columns={'nn': 'nn_mse'}), on='time', how='inner')
    D = D.dropna(subset=['linear', 'recurrence']).sort_values('time')
    o = D.obs.to_numpy()
    t = pd.to_datetime(D.time)
    blk = ((t - t.min()) / pd.Timedelta(days=27)).astype(int).to_numpy()
    nb = blk.max() + 1
    rng = np.random.default_rng(0)
    W = [np.ones(len(o))] + [np.bincount(rng.integers(0, nb, nb), minlength=nb)[blk].astype(float) for _ in range(NBOOT)]
    cols = {'B0 (S loss)': 'nn', 'B0, MSE loss': 'nn_mse', 'Linear, set 2a': 'linear', '27-day recurrence': 'recurrence'}
    pairs = [('B0 (S loss)', 'Linear, set 2a'), ('B0, MSE loss', 'Linear, set 2a'), ('B0 (S loss)', 'B0, MSE loss'),
             ('B0 (S loss)', '27-day recurrence')]
    rows = []
    for v in (500, 600):
        pos = o >= v
        stat = {}
        for name, c in cols.items():
            p = D[c].to_numpy()
            stat[name] = np.array([[roc_auc_score(pos, p, sample_weight=w), best_tss(p, pos, w)] for w in W])
        for a, b in pairs:
            d = stat[a] - stat[b]
            for k, metric in enumerate(('auc', 'tss_best')):
                lo, hi = np.percentile(d[1:, k], [2.5, 97.5])
                rows.append(dict(threshold=v, metric=metric, a=a, b=b, a_value=stat[a][0, k], b_value=stat[b][0, k],
                                 diff=d[0, k], ci_lo=lo, ci_hi=hi, frac_le0=float(np.mean(d[1:, k] <= 0))))
        print(f'{v} km/s done', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f'{B0_RUN}/skill_bootstrap.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.round(3).to_string(index=False))
