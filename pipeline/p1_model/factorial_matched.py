"""What the solver adds (first review, item 1; second review, items 1-2): the factorial variants of B0 on the matched
development test hours, with the headline convention (mean of the seed-42 and seed-43 predictions, pooled over the five
test batches) and paired 27-day block-bootstrap 95 % intervals of ΔS against B0 (as b0_matched_S.py).

Variants (runs/sweep4, identical inputs, loss and training except as named):
  F_ball     network + ballistic link (each parcel at its own speed plus the residual acceleration, no interaction)
  F_lag      network + fixed 96-h lag at the source longitude (the alignment of the linear comparator)
  F_lin      linear map (no hidden layers) + solver
  F_lin_lag  linear map + fixed 96-h lag
  G_nam      neural additive model with 3 pairwise terms + solver (2026-10-18; boundary.NAMHead)
Common hours: where every variant, B0, the linear comparator and the recurrence exist.

  python pipeline/p1_model/factorial_matched.py
Output: runs/sweep4/factorial_matched.csv
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from b0_skill_metrics import two_seed
from eval_matched_baselines import ace_hourly, metrics
from frozen_model import B0_RUN
from heldout_scores import boot_dS

SWEEP = B0_RUN.rsplit('/', 1)[0]
VARIANTS = [('B0', 'B0: network + solver'), ('F_ball', 'network + ballistic'), ('F_lag', 'network + fixed 96-h lag'),
            ('F_lin', 'linear + solver'), ('F_lin_lag', 'linear + fixed 96-h lag'),
            ('G_nam', 'additive model + 3 pairs (NAM) + solver')]

if __name__ == '__main__':
    ace = ace_hourly()
    D = None
    for cfg, _ in VARIANTS:
        T = two_seed(f'{SWEEP}/{cfg}', ace)
        cols = ['time', 'nn'] if D is not None else ['time', 'obs', 'linear', 'recurrence', 'nn']
        T = T[cols].rename(columns={'nn': cfg})
        D = T if D is None else D.merge(T, on='time', how='inner')
    D = D.dropna().sort_values('time')
    o = D.obs.to_numpy()
    t = pd.to_datetime(D.time)
    blk = ((t - t.min()) / pd.Timedelta(days=27)).astype(int).to_numpy()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    e0 = (D.B0.to_numpy() - o) ** 2
    S0 = metrics(o, D.B0.to_numpy())['S']
    rows = []
    for cfg, lab in VARIANTS + [('linear', 'linear comparator (6 AIA + 2 HMI, MSE, 4-d lag)'), ('recurrence', '27-day recurrence')]:
        p = D[cfg].to_numpy()
        m = metrics(o, p)
        row = dict(variant=cfg, label=lab, n=len(o), S=m['S'], rmse=m['rmse'], trmse=m['trmse'], bias=m['bias'], r=m['r'],
                   amp=m['amp'], tss500=m['tss500'], tss600=m['tss600'],
                   auc500=roc_auc_score(o >= 500, p), auc600=roc_auc_score(o >= 600, p), dS=m['S'] - S0)
        if cfg != 'B0':
            ds = boot_dS(o, (p - o) ** 2, e0, blk, 1000, dev)
            row.update(dS_lo=np.percentile(ds, 2.5), dS_hi=np.percentile(ds, 97.5))
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(f'{SWEEP}/factorial_matched.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.drop(columns='label').round(3).to_string(index=False))
