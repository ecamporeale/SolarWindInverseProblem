"""Back-mapped recurrence on the NN1 grid (per Carrington longitude and hour): an NN1 input channel.

Every measured ACE hour (t_obs, speed v, ACE Carrington longitude) is traced ballistically to its source:
t_s = t_obs - (215 - 21.5) Rs / v, source longitude = ACE Carrington longitude at t_s (as knn_backmap_test.py).
Cell (model longitude L_j = j * 360/128, hour t): mean speed of the hours whose source fell within +-5 deg of L_j at
t_s in [t - 27.2753 d - 5 d, t - 27.2753 d + 5 d] (the previous passage). Those hours were observed at
t_s + travel time <= t - 15.7 d, i.e. long before t (checked), so the channel is causal.
Outputs: data/nn1_dataset/bm_recurrence_nphi128.npy (km/s; 400 where no data), bm_recurrence_valid_nphi128.npy (1/0)
"""
import os

import numpy as np
import pandas as pd

DS = os.environ.get('NN1_DATASET', 'data/nn1_dataset')                 # dataset dir (times.csv in, outputs out)
OBS = os.environ.get('OBS_FILE', 'data/preprocessed_window/multi_spacecraft_obs_20110101_20211231T235959.csv')


def ace_hourly():
    o = pd.read_csv(OBS, parse_dates=['Time'])
    o = o[o.Spacecraft_ID == 'ACE']
    return o.set_index('Time')['Speed'].sort_index().resample('1h').mean()

NPHI, DLON, ROT_D, WIN_D = 128, 5.0, 27.2753, 5.0      # +-5 d: covers the visible disk (+-66 deg)
D_KM = (215.0 - 21.5) * 695700.0

times = pd.to_datetime(pd.read_csv(f'{DS}/times.csv', usecols=['time'])['time'])
vh = ace_hourly().dropna()
o = pd.read_csv(OBS, parse_dates=['Time'])
lon = o[o.Spacecraft_ID == 'ACE'].set_index('Time')['HC_lon'].resample('1h').mean()
unw = pd.Series(np.unwrap(np.deg2rad(lon.interpolate(limit=48).to_numpy())), index=lon.index).dropna()
ts = vh.index - pd.to_timedelta(D_KM / vh.to_numpy(), unit='s')
ls = np.rad2deg(np.interp(ts.asi8, unw.index.asi8, unw.to_numpy())) % 360.0
tsn, vv, tobs = ts.asi8, vh.to_numpy(), vh.index.asi8
tq = times.to_numpy().astype('datetime64[ns]').astype(np.int64) - int(ROT_D * 86400e9)
lo, hi = tq - int(WIN_D * 86400e9), tq + int(WIN_D * 86400e9)
BM = np.full((NPHI, len(times)), 400.0, np.float32); VAL = np.zeros((NPHI, len(times)), np.uint8)
latest_obs = np.full(len(times), np.iinfo(np.int64).min)
for j in range(NPHI):
    L = j * 360.0 / NPHI
    m = np.abs((ls - L + 180.0) % 360.0 - 180.0) <= DLON
    order = np.argsort(tsn[m]); t_j, v_j, o_j = tsn[m][order], vv[m][order], tobs[m][order]
    cs = np.concatenate([[0.0], np.cumsum(v_j)])
    a, b = np.searchsorted(t_j, lo), np.searchsorted(t_j, hi)
    cnt = b - a
    ok = cnt > 0
    BM[j, ok] = (cs[b[ok]] - cs[a[ok]]) / cnt[ok]; VAL[j, ok] = 1
    omax = np.maximum.accumulate(o_j) if len(o_j) else o_j                      # latest observation time used
    latest_obs[ok] = np.maximum(latest_obs[ok], omax[b[ok] - 1])
t_ns = times.to_numpy().astype('datetime64[ns]').astype(np.int64)
used = latest_obs > np.iinfo(np.int64).min
margin_d = (t_ns[used] - latest_obs[used]) / 86400e9
assert margin_d.min() > 0, 'non-causal back-mapped value'
np.save(f'{DS}/bm_recurrence_nphi128.npy', BM)
np.save(f'{DS}/bm_recurrence_valid_nphi128.npy', VAL)
print(f'cells with data: {100 * VAL.mean():.1f}%; speed p5/p50/p95 = {np.percentile(BM[VAL == 1], [5, 50, 95]).round(0)}; '
      f'min (t - latest observation used) = {margin_d.min():.1f} d')
