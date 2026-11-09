"""Regression check: the frozen model B0 (frozen_model.B0_RUN), seed 42, fold 1, run through the current code on its
test chunks must reproduce the stored test predictions (written by eval_matched_baselines.py; differences of a few
1e-5 km/s come from GPU arithmetic). First written for the split refactor (2026-10-15); B0 = sweep 4 since 2026-10-16.

  python checks/check_b5_regression.py <gpu index>"""
import sys
import numpy as np, pandas as pd, torch
from frozen_model import B0_RUN
from plot_best_checkpoint import chunks_of, load_fold, scored_predictions
dev = torch.device(f'cuda:{sys.argv[1]}')
F = load_fold(1, f'{B0_RUN}/seed42/fold1', dev)
d = pd.concat([scored_predictions(F, c) for c in chunks_of(F, 'test')], ignore_index=True)
ref = pd.read_csv(f'{B0_RUN}/seed42/fold1/test_predictions.csv', parse_dates=['time'])
m = d.merge(ref, on='time', suffixes=('', '_ref'))
print(len(d), len(ref), len(m), 'max |pred diff|', np.abs(m.pred - m.pred_ref).max(), 'max |obs diff|', np.abs(m.obs - m.obs_ref).max())
