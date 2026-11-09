"""Per-fold summary of a training sweep from its train.log files (the table read by the ablation and loss-comparison
figures and by paper3_tables.py). Written 2026-10-16; runs/sweep3/results_by_fold.csv had been assembled the same way.

  python pipeline/p1_model/summarize_sweep.py runs/sweep4
Output: <sweep>/results_by_fold.csv (cfg, seed, fold, done, best_ep, valS, testS, rmse, trmse, bias)
"""
import re
import sys
from pathlib import Path

import pandas as pd

sweep = Path(sys.argv[1])
rows = []
for log in sorted(sweep.glob('*/seed*/fold*/train.log')):
    cfg, seed, fold = log.parts[-4], int(log.parts[-3][4:]), int(log.parts[-2][4:])
    t = log.read_text()
    b = re.search(r'Loaded best checkpoint: epoch=(\d+) val_score=([\d.e+-]+)', t)
    s = re.search(r'Combined score S \(test grid\)\s*:\s*([\d.]+) km/s\s+\(RMSE ([\d.]+), TRMSE ([\d.]+), bias ([+-][\d.]+)', t)
    rows.append(dict(cfg=cfg, seed=seed, fold=fold, done=bool(b and s), best_ep=int(b.group(1)) if b else None,
                     valS=round(float(b.group(2)), 5) if b else None, testS=float(s.group(1)) if s else None,
                     rmse=float(s.group(2)) if s else None, trmse=float(s.group(3)) if s else None,
                     bias=float(s.group(4)) if s else None))
R = pd.DataFrame(rows).sort_values(['cfg', 'seed', 'fold'])
R.to_csv(sweep / 'results_by_fold.csv', index=False)
print(R.groupby('cfg')[['done', 'valS', 'testS', 'rmse', 'bias']].agg({'done': 'sum', 'valS': 'mean', 'testS': 'mean',
                                                                     'rmse': 'mean', 'bias': 'mean'}).round(2).to_string())
