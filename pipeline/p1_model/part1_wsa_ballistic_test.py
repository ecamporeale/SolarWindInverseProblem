"""Part I comparison (2026-10-24), third version: WSA 4.5 + ballistic mapping over B0's test chunks.

Forward ballistic mapping as Arge & Pizzo (2000), for the scored test hours of B0 (runs/sweep4/B0/seed42/fold<k>/
test_predictions.csv, the hours of the other two versions): every hour t_s one parcel leaves 21.5 Rs from the Carrington
longitude facing ACE (in the synodic frame, the parcel that reaches ACE), at WSA's 1-AU-equivalent speed there in the map
current at t_s (latest map at or before t_s; part1_wsa_profiles.py, ACE latitude, 128 longitudes, periodic
interpolation), travelling radially at constant speed to ACE's distance. Parcels overtaken by later, faster ones are
dropped (the faster replaces the slower); arrival speeds are interpolated linearly to the scored hours. No stream
interaction and no acceleration: WSA's speed equation already gives the asymptotic speed. (A backward root search was
tried first: up to 17 self-consistent departures per hour, mostly from map-to-map changes, so it was not used.) ACE
positions: data/ace_dev/ACE_speed_lon_lat_hr.txt.

  python pipeline/p1_model/part1_wsa_ballistic_test.py
Output: runs/part1_wsa/ballistic_test.csv  time, fold, chunk, wsa_ballistic, dt_h, map_age_h, gap_h
"""
import glob

import numpy as np
import pandas as pd

RSUN_KM, AU_RS = 695700.0, 215.032
H = pd.Timedelta(hours=1)

if __name__ == '__main__':
    W = np.load('runs/part1_wsa/wsa45_profiles.npz', allow_pickle=True)
    mt = pd.DatetimeIndex(W['map_time']).values; V = W['v1au'].astype(float); LON = W['lon']
    ace = pd.read_csv('data/ace_dev/ACE_speed_lon_lat_hr.txt', parse_dates=['Time']).set_index('Time')
    rows = []
    for f in sorted(glob.glob('runs/sweep4/B0/seed42/fold*/test_predictions.csv')):
        fold = int(f.split('fold')[-1].split('/')[0])
        d = pd.read_csv(f, parse_dates=['time'])[['time', 'chunk']]
        d['fold'] = fold
        rows.append(d)
    D = pd.concat(rows, ignore_index=True)
    T = pd.DatetimeIndex(D.time)
    r_ace = ace['HC_radius'].astype(float).interpolate(limit_area='inside').reindex(T).to_numpy() * AU_RS
    dist_km = (r_ace - 21.5) * RSUN_KM
    # forward mapping (Arge & Pizzo 2000): one parcel per hour leaves 21.5 Rs from the longitude facing ACE then (in the
    # synodic frame the parcel that will reach ACE: φ_s = φ_ACE(t_s)), at the WSA speed of the map current at t_s, and
    # arrives at t_s + dist / v. A parcel overtaken by a later, faster one (later departure, earlier arrival) is dropped;
    # arrival speeds are interpolated linearly to the scored hours.
    ts = pd.date_range(T.min().floor('h') - pd.Timedelta(hours=300), T.max(), freq='h')
    lon0 = np.deg2rad(ace['HC_lon'].astype(float)).dropna()
    lon_u = pd.Series(np.unwrap(lon0.to_numpy()), index=lon0.index)                 # unwrap the measured hours only
    lon_s = lon_u.reindex(lon_u.index.union(ts)).interpolate('time', limit_area='inside').reindex(ts).to_numpy()
    r_s = ace['HC_radius'].astype(float).interpolate(limit_area='inside').reindex(ts).bfill().ffill().to_numpy() * AU_RS
    k = np.searchsorted(mt, ts.values, side='right') - 1
    phi = lon_s % (2 * np.pi)
    x = phi / (2 * np.pi) * LON.size - 0.5
    i0 = np.floor(x).astype(int); w = x - i0; kk = np.maximum(k, 0)
    v_dep = (1 - w) * V[kk, i0 % LON.size] + w * V[kk, (i0 + 1) % LON.size]
    v_dep[(k < 0) | ~np.isfinite(lon_s)] = np.nan
    age = (ts.values - mt[kk]) / np.timedelta64(1, 'h')
    arr = ts.values.astype('datetime64[ns]').astype(np.int64) / 3.6e12 + (r_s - 21.5) * RSUN_KM / v_dep / 3600.0   # hours
    keep_t, keep_v, keep_age, dropped = [], [], [], 0
    for ta, va, ag in zip(arr, v_dep, age):
        if not np.isfinite(ta):
            continue
        while keep_t and keep_t[-1] >= ta:              # earlier, slower parcels overtaken by this one
            keep_t.pop(); keep_v.pop(); keep_age.pop(); dropped += 1
        keep_t.append(ta); keep_v.append(va); keep_age.append(ag)
    keep_t, keep_v, keep_age = map(np.array, (keep_t, keep_v, keep_age))
    th = T.values.astype('datetime64[ns]').astype(np.int64) / 3.6e12
    D['wsa_ballistic'] = np.interp(th, keep_t, keep_v, left=np.nan, right=np.nan)
    D['dt_h'] = (r_ace - 21.5) * RSUN_KM / D.wsa_ballistic.to_numpy() / 3600.0
    D['map_age_h'] = np.interp(th, keep_t, keep_age)
    D['gap_h'] = np.diff(keep_t, append=np.nan)[np.clip(np.searchsorted(keep_t, th) - 1, 0, len(keep_t) - 1)]
    D.to_csv('runs/part1_wsa/ballistic_test.csv', index=False)
    print(f'{len(D)} hours; finite {np.isfinite(D.wsa_ballistic).mean():.3f}; parcels dropped as overtaken {dropped} of '
          f'{np.isfinite(arr).sum()}; travel time {D.dt_h.median():.0f} h median; arrival gaps > 6 h at {(D.gap_h > 6).mean():.3f} '
          f'of hours; map older than 48 h at departure {(D.map_age_h > 48).mean():.3f}')
