"""Matched-hours table of B0 (development test batches) with the in-house baselines, seed mean of the NN
(column nn; nn_mse = the same for B0 trained with the MSE loss, <B0_RUN>_mse).
Built from the stored test_predictions.csv of <B0_RUN>/seed{42,43} (frozen_model.B0_RUN) (no GPU), cached in
figures/part1/_cache/b0_matched_hourly.csv."""
from pathlib import Path

import pandas as pd
import torch

from eval_matched_baselines import ace_hourly, matched_table
from frozen_model import B0_RUN

CACHE = Path('figures/part1/_cache/b0_matched_hourly_v2.csv')    # v2: retrained B0 (Carrington Ω) + nn_mse


def b0_hourly():
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=['time'])
    ace = ace_hourly()
    T = [matched_table(f'{B0_RUN}/seed{s}', torch.device('cpu'), ace) for s in (42, 43)]
    D = T[0].merge(T[1][['time', 'nn']], on='time', suffixes=('_s42', '_s43'))
    D['nn'] = 0.5 * (D.nn_s42 + D.nn_s43)
    M = [matched_table(f'{B0_RUN}_mse/seed{s}', torch.device('cpu'), ace) for s in (42, 43)]
    M = M[0][['time', 'nn']].merge(M[1][['time', 'nn']], on='time')
    D = D.merge(M.assign(nn_mse=0.5 * (M.nn_x + M.nn_y))[['time', 'nn_mse']], on='time', how='left')
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(CACHE, index=False)
    return D
