"""Part I comparison (2026-10-24), headline version: WSA 4.5 + HUXt run exactly over B0's test chunks, as B0 runs there.

B0 is evaluated on its test batches chunk by chunk: the solver runs from the chunk start with a boundary that changes
every hour (NN1 at each hour from the images available then), the first 150 h are warm-up and not scored, and the
scored hours are those of runs/sweep4/B0/seed42/fold<k>/test_predictions.csv (same chunks for seed 43). Here the same
chunks are run with WSA 4.5 + HUXt: time-dependent inner boundary at every hour = the latest WSA 4.5 map at or before
that hour (profiles from part1_wsa_profiles.py: ACE latitude, 128 Carrington longitudes), decelerated from the
1-AU-equivalent WSA speed to 21.5 Rs with HUXt's own huxt_inputs.map_v_inwards, propagated with HUXt 5.0.3
(huxt_inputs.set_time_dependent_boundary, 1-D along the Sun–Earth line, synodic, r = 21.5 .. 240 Rs), run start =
first scored hour − 150 h (the same warm-up), sampled hourly at ACE's heliocentric distance. A map older than 2 days at
an hour is still used (held); the hours affected are flagged.
Run with the `huxt` environment.

  $PY_H pipeline/p1_model/part1_wsa_huxt_test.py [--workers 40]
  2026-10-26: [--profiles F --col C --windows CSV --out F] for WSA+ over its test rotations (part1_wsaplus_*.py)
Output: runs/part1_wsa/huxt_test.csv  time, fold, chunk, wsa_huxt, map_age_h
"""
import argparse
import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd

WARM = pd.Timedelta(hours=150)
H = pd.Timedelta(hours=1)
ACE = 'data/ace_dev/ACE_speed_lon_lat_hr.txt'
RSUN_AU = 215.032
P = {}
PROFILES, COL = 'runs/part1_wsa/wsa45_profiles.npz', 'wsa_huxt'


def _init(profiles=PROFILES):
    W = np.load(profiles, allow_pickle=True)
    P['t'] = pd.DatetimeIndex(W['map_time']); P['v'] = W['v1au']; P['lon'] = W['lon']
    a = pd.read_csv(ACE, parse_dates=['Time']).set_index('Time')
    P['r'] = a['HC_radius'].astype(float).interpolate(limit_area='inside') * RSUN_AU


def run_chunk(args):
    fold, cid, t_first, t_last = args
    import astropy.units as u
    from astropy.time import Time
    from huxt import huxt as Hx, huxt_inputs as Hin
    t0 = t_first - WARM
    hours = pd.date_range(t0, t_last, freq='h')
    k = np.searchsorted(P['t'].values, hours.values, side='right') - 1
    if (k < 0).any():
        return None
    age = (hours - P['t'][k]) / H
    lon = P['lon'] * u.rad
    vgrid = np.empty((lon.size, len(hours)))
    for j, kk in enumerate(np.unique(k)):
        v_in, _ = Hin.map_v_inwards(P['v'][kk] * u.km / u.s, 215 * u.solRad, lon, 21.5 * u.solRad)
        vgrid[:, k == kk] = v_in.value[:, None]
    simtime = (len(hours) + 2) / 24 * u.day
    model = Hin.set_time_dependent_boundary(vgrid * u.km / u.s, Time(hours.to_pydatetime()).mjd, t0.to_pydatetime(), simtime,
                                            r_min=21.5 * u.solRad, r_max=240 * u.solRad, dt_scale=4, frame='synodic',
                                            lon_out=0 * u.rad, track_cmes=False)
    model.solve([])
    th = model.time_out.to(u.s).value / 3600.0
    r = model.r.to(u.solRad).value
    V = model.v_grid[:, :, 0].value
    sc = pd.date_range(t_first, t_last, freq='h')
    tau = (sc - t0) / H
    ra = P['r'].reindex(sc).to_numpy()
    out = np.full(len(sc), np.nan)
    for n, (tt, rr) in enumerate(zip(tau, ra)):
        if np.isfinite(rr):
            j0 = np.searchsorted(r, rr) - 1
            vt = [np.interp(tt, th, V[:, j0]), np.interp(tt, th, V[:, j0 + 1])]
            out[n] = np.interp(rr, r[j0:j0 + 2], vt)
    return pd.DataFrame({'time': sc, 'fold': fold, 'chunk': cid, P.get('col', COL): out,
                         'map_age_h': np.asarray(age)[hours.get_indexer(sc)]})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--profiles', default=PROFILES, help='boundary profiles (map_time, lon, v1au), e.g. runs/part1_wsa/wsaplus_profiles.npz')
    ap.add_argument('--col', default=COL, help='output column')
    ap.add_argument('--windows', default=None, help='csv with t_first, t_last (scored span; warm-up 150 h before) instead of the test chunks')
    ap.add_argument('--out', default='runs/part1_wsa/huxt_test.csv')
    a = ap.parse_args()
    jobs = []
    if a.windows:                                                     # e.g. WSA+'s test rotations (2026-10-26)
        Wn = pd.read_csv(a.windows, parse_dates=['t_first', 't_last'])
        jobs = [(-1, int(r.cr), r.t_first, r.t_last) for r in Wn.itertuples()]
    for f in ([] if a.windows else sorted(glob.glob('runs/sweep4/B0/seed42/fold*/test_predictions.csv'))):
        fold = int(f.split('fold')[-1].split('/')[0])
        d = pd.read_csv(f, parse_dates=['time'])
        for cid, g in d.groupby('chunk'):
            jobs.append((fold, int(cid), g.time.min(), g.time.max()))
    print(f'{len(jobs)} test chunks', flush=True)
    P['col'] = a.col

    def _init_all(profiles=a.profiles, col=a.col):
        _init(profiles); P['col'] = col
    with Pool(min(a.workers, len(jobs)), initializer=_init_all) as pool:
        res = pool.map(run_chunk, jobs, chunksize=1)
    bad = [j for j, r in zip(jobs, res) if r is None]
    D = pd.concat([r for r in res if r is not None], ignore_index=True)
    D.to_csv(a.out, index=False)
    print(f'saved {a.out}: {len(D)} hours, finite {np.isfinite(D[a.col]).mean():.3f}; '
          f'chunks without a map: {bad}; hours with map older than 48 h: {(D.map_age_h > 48).mean():.3f}', flush=True)
