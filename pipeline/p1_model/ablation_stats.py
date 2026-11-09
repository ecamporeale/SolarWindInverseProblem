"""Are the one-change ablations of B0 distinguishable from zero? (second review, item 4, 2026-10-17)

Per variant, paired with B0 on the same (seed, fold): ΔS = S(variant) − S(B0) on the test and validation batches.
Reported: mean ± standard error over the 10 runs (as Figure 9a); the conservative fold-level test (seeds averaged per
fold, 5 paired differences, two-sided t-test with 4 degrees of freedom), since the two seeds of a fold share their
data; and the number of folds whose fold-mean ΔS has the sign of the overall mean.

  python pipeline/p1_model/ablation_stats.py runs/sweep4
Output: <sweep>/ablation_stats.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sweep = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/sweep4')
R = pd.read_csv(sweep / 'results_by_fold.csv')
R = R[R.done]
base = R[R.cfg == 'B0'].set_index(['seed', 'fold'])
rows = []
for cfg, g in R[R.cfg != 'B0'].groupby('cfg'):
    g = g.set_index(['seed', 'fold'])
    for split, col in (('test', 'testS'), ('val', 'valS')):
        d = (g[col] - base[col].reindex(g.index)).dropna()
        f = d.groupby(level='fold').mean()
        t, p = stats.ttest_1samp(f, 0.0) if len(f) > 1 else (np.nan, np.nan)
        rows.append(dict(cfg=cfg, split=split, n_runs=len(d), mean=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)),
                         n_folds=len(f), fold_se=f.std(ddof=1) / np.sqrt(len(f)), t=t, p=p,
                         folds_same_sign=int((np.sign(f) == np.sign(d.mean())).sum())))
S = pd.DataFrame(rows)
S.to_csv(sweep / 'ablation_stats.csv', index=False)
pd.set_option('display.width', 200)
print(S.round(3).to_string(index=False))
