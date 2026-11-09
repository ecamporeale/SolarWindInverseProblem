"""Part I case studies, stage 1 of 3: the development out-of-fold series on OMNI time, for the PROSWIN high-speed-stream
event protocol (the same protocol as Part II, PAPER_5 §4; stage 2: dev_events_eval.py, windnet env; stage 3:
dev_cases_select.py).

Series: B0 (mean of the seed-42 and seed-43 test predictions of the frozen model (frozen_model.B0_RUN), development test batches), the linear
set-2a comparator and the 27-day recurrence, on B0's matched hours (eval_matched_baselines.matched_table, as in
PAPER_3 §5.3). Each is moved from the ACE label axis to OMNI time with part2_omni.to_omni_time (delay from the series'
own speed). Observation: OMNI hourly, all hours (PROSWIN setting). Rows: OMNI hours where all four are finite.

  python pipeline/p1_model/dev_events_series.py [--runs <B0_RUN>] [--out runs/dev_events]
Output: runs/dev_events/series.csv (index time; obs, NN, Linear, Recurrence), runs/dev_events/ace_axis.csv
        (--with-wsa --out runs/dev_events_wsa: + column WSA+HUXt; same rows)
"""
import argparse
import os

import pandas as pd
import torch

from eval_matched_baselines import ace_hourly, matched_table
from part2_omni import load_omni, to_omni_time
from frozen_model import B0_RUN


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=B0_RUN, help='directory with seed42/, seed43/ (fold*/test_predictions.csv)')
    ap.add_argument('--out', default='runs/dev_events')
    ap.add_argument('--with-wsa', action='store_true',
                    help='also WSA 4.5 + HUXt (runs/part1_wsa/matched.csv, column wsa_huxt; 2026-10-25), on the same hours')
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    ace = ace_hourly()
    T = [matched_table(f'{a.runs}/seed{s}', torch.device('cpu'), ace) for s in (42, 43)]
    D = T[0].merge(T[1][['time', 'nn']], on='time', suffixes=('_s42', '_s43'))
    D['nn'] = 0.5 * (D.nn_s42 + D.nn_s43)
    D = D.dropna(subset=['linear', 'recurrence']).set_index('time').sort_index()
    D.to_csv(f'{OUT}/ace_axis.csv')
    omni = load_omni('none')
    cols = {'obs': omni, 'NN': to_omni_time(D.nn), 'Linear': to_omni_time(D.linear), 'Recurrence': to_omni_time(D.recurrence)}
    if a.with_wsa:
        W = pd.read_csv('runs/part1_wsa/matched.csv', parse_dates=['time']).set_index('time').wsa_huxt
        assert W.reindex(D.index).notna().all(), 'WSA + HUXt missing on some matched hours: the row set would change'
        cols['WSA+HUXt'] = to_omni_time(W.reindex(D.index))
    S = pd.DataFrame(cols)
    S = S[(S.index >= D.index[0]) & (S.index <= D.index[-1] + pd.Timedelta(hours=2))].dropna()
    S.to_csv(f'{OUT}/series.csv')
    print(f'ACE axis: {len(D)} matched hours; OMNI axis: {len(S)} h, {S.index[0]} .. {S.index[-1]}')
