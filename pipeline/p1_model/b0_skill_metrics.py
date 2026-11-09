"""Bias-aware and threshold-free skill on B0's matched development hours (reviewer comment 2, 2026-10-17).

The combined score S weights errors by the observed speed, which rewards over-forecasting; TSS at fixed thresholds also
rewards an upward bias. To see how much skill survives once bias is removed, for each method on the matched hours
(eval_matched_baselines.matched_table, as PAPER_3 §5.3 / manuscript Table 2):
  bias, RMSE, debiased RMSE (prediction minus its own mean bias), r, amplitude ratio σ_pred/σ_obs;
  for fast wind ≥ 500 and ≥ 600 km/s:
    TSS raw; TSS of the debiased prediction; best TSS over prediction thresholds 300-800 km/s (each model's own threshold);
    ROC AUC (threshold-free); HSS raw.
Methods: B0 and B0_mse (each the mean of seeds 42 and 43), linear set-2a comparator, 27-day recurrence, climatology.

  python pipeline/p1_model/b0_skill_metrics.py
Output: <B0_RUN>/skill_metrics.csv
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from eval_matched_baselines import ace_hourly, matched_table
from frozen_model import B0_RUN


def _bf(o, p, v):
    from sklearn.metrics import roc_auc_score, roc_curve
    pos = o >= v
    tss = np.mean(p[pos] >= v) - np.mean(p[~pos] >= v)
    fpr, tpr, thr = roc_curve(pos, p); k = np.argmax(tpr - fpr)
    return tss, roc_auc_score(pos, p), tpr[k] - fpr[k], thr[k]


def two_seed(run, ace):
    T = [matched_table(f'{run}/seed{s}', torch.device('cpu'), ace) for s in (42, 43)]
    D = T[0].merge(T[1][['time', 'nn']], on='time', suffixes=('_s42', '_s43'))
    D['nn'] = 0.5 * (D.nn_s42 + D.nn_s43)
    return D


def contingency(p, o, thr_p, thr_o):
    hit = np.sum((p >= thr_p) & (o >= thr_o)); miss = np.sum((p < thr_p) & (o >= thr_o))
    fa = np.sum((p >= thr_p) & (o < thr_o)); cn = np.sum((p < thr_p) & (o < thr_o))
    return hit, miss, fa, cn


def tss(p, o, thr_p, thr_o):
    hit, miss, fa, cn = contingency(p, o, thr_p, thr_o)
    return hit / (hit + miss) - fa / (fa + cn)


def hss(p, o, thr):
    hit, miss, fa, cn = contingency(p, o, thr, thr)
    den = (hit + miss) * (miss + cn) + (hit + fa) * (fa + cn)
    return 2.0 * (hit * cn - miss * fa) / den


if __name__ == '__main__':
    ace = ace_hourly()
    D = two_seed(B0_RUN, ace)
    M = two_seed(B0_RUN.replace('/B0', '/B0_mse'), ace)
    D = D.merge(M[['time', 'nn']].rename(columns={'nn': 'nn_mse'}), on='time', how='inner')
    D = D.dropna(subset=['linear', 'recurrence'])
    o = D.obs.to_numpy()
    rows = []
    for name, col in (('B0 (S loss)', 'nn'), ('B0, MSE loss', 'nn_mse'), ('Linear, set 2a', 'linear'),
                      ('27-day recurrence', 'recurrence'), ('Climatology', 'climatology')):
        p = D[col].to_numpy()
        b = float(np.mean(p - o)); pd_ = p - b
        row = dict(method=name, n=len(o), bias=b, rmse=float(np.sqrt(np.mean((p - o) ** 2))),
                   rmse_debiased=float(np.sqrt(np.mean((pd_ - o) ** 2))), r=float(np.corrcoef(p, o)[0, 1]),
                   amp=float(np.std(p) / np.std(o)))
        for v in (500, 600):
            const = np.std(p) == 0
            row[f'tss{v}'] = tss(p, o, v, v)
            row[f'tss{v}_debiased'] = tss(pd_, o, v, v)
            row[f'tss{v}_best'] = 0.0 if const else max(tss(p, o, th, v) for th in np.arange(300, 801, 5))
            row[f'auc{v}'] = 0.5 if const else roc_auc_score(o >= v, p)
            row[f'hss{v}'] = 0.0 if const else hss(p, o, v)
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(f'{B0_RUN}/skill_metrics.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.round(3).to_string(index=False))
