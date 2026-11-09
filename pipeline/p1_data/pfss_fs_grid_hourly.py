"""PFSS expansion factor on the NN1 longitude grid, HOURLY: every AIA hour uses the most recent GONG mrzqs map at or
before it (GONG cadence: 6 h until Oct 2012, ~1 h afterwards; maps at ~:04/:14). Same solver and tracing as
pfss_fs_grid.py (rss 2.5, nrho 30, field lines traced down from the source surface at the 128 model longitudes and
Earth's B0 at the map time; non-open lines take the value of the nearest open longitude).
Output: data/nn1_dataset/pfss_log10fs_hourly_nphi128.npy float32 [128, Nt_hourly] (aligned with times.csv);
        data/nn1_dataset/pfss_log10fs_hourly_maps.csv (map, map_time, status, n_open);
        data/nn1_dataset/pfss_log10fs_hourly_age.npy (age of the map used, hours)
"""
import glob
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from data_paths import SOLAR

warnings.filterwarnings('ignore')
GONG = str(SOLAR / 'GONG_MAG')
DS = os.environ.get('NN1_DATASET', 'data/nn1_dataset')     # dataset dir (times.csv in, outputs out)
NPHI, RSS, NRHO = 128, 2.5, 30
LONS = np.arange(NPHI) * 360.0 / NPHI


def map_fs(args):
    path, tmap = args
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import sun
    import sunkit_magex.pfss as pfss
    try:
        m = sunpy.map.Map(path).resample([360, 180] * u.pix)
        pout = pfss.pfss(pfss.Input(m, NRHO, RSS))
        b0 = float(sun.B0(tmap).to_value(u.deg))
        s = SkyCoord(LONS * u.deg, np.full(NPHI, b0) * u.deg, (RSS - 0.01) * u.R_sun, frame=pout.coordinate_frame)
        fls = pfss.tracing.PerformanceTracer().trace(s, pout)
        ef = np.asarray(fls.expansion_factors, float)
        ok = np.array([fl.is_open for fl in fls]) & np.isfinite(ef) & (ef > 0)
        if not ok.any():
            return path, None, 'no_open', 0
        out = np.full(NPHI, np.nan); out[ok] = np.log10(ef[ok])
        idx = np.flatnonzero(ok)
        for j in np.flatnonzero(~ok):
            dd = np.abs((idx - j + NPHI // 2) % NPHI - NPHI // 2)
            out[j] = out[idx[dd.argmin()]]
        return path, out.astype(np.float32), 'ok', int(ok.sum())
    except Exception as e:
        return path, None, f'error: {type(e).__name__}: {e}'[:160], 0


if __name__ == '__main__':
    from multiprocessing import Pool
    times = pd.to_datetime(pd.read_csv(f'{DS}/times.csv', usecols=['time'])['time'])
    t_lo, t_hi = times.iloc[0] - pd.Timedelta(days=2), times.iloc[-1]
    maps = []
    for y in range(t_lo.year, t_hi.year + 1):
        for f in glob.glob(f'{GONG}/{y}/*/mrzqs*.fits.gz'):
            mm = re.search(r'mrzqs(\d{6})t(\d{4})', os.path.basename(f))
            if mm:
                t = pd.Timestamp('20' + mm.group(1) + mm.group(2))
                if t_lo <= t <= t_hi:
                    maps.append((t, f))
    M = pd.DataFrame(maps, columns=['map_time', 'path']).sort_values('map_time').drop_duplicates('map_time').reset_index(drop=True)
    j = np.searchsorted(M.map_time.to_numpy(), times.to_numpy(), side='right') - 1     # most recent map <= AIA hour
    used = np.unique(j[j >= 0])
    print(f'{len(M)} maps in window; {len(used)} used by {len(times)} AIA hours', flush=True)
    t0 = time.time(); res = {}
    with Pool(48) as pool:
        for i, (p, v, st, no) in enumerate(pool.imap_unordered(map_fs, [(M.path[k], M.map_time[k]) for k in used], chunksize=8), 1):
            res[p] = (v, st, no)
            if i % 5000 == 0:
                print(f'  {i}/{len(used)} maps, {(time.time() - t0) / 60:.1f} min', flush=True)
    # fall back to the previous successful map when a map failed
    vals, last = {}, None
    for k in range(len(M)):
        r = res.get(M.path[k])
        if r is not None and r[0] is not None:
            last = k
        vals[k] = last
    G = np.full((NPHI, len(times)), np.nan, np.float32); age = np.full(len(times), np.nan, np.float32)
    for i, k in enumerate(j):
        if k < 0 or vals.get(k) is None:
            continue
        kk = vals[k]
        G[:, i] = res[M.path[kk]][0]; age[i] = (times.iloc[i] - M.map_time[kk]) / pd.Timedelta(hours=1)
    np.save(f'{DS}/pfss_log10fs_hourly_nphi128.npy', G)
    np.save(f'{DS}/pfss_log10fs_hourly_age.npy', age)
    st = pd.DataFrame([(M.path[k], M.map_time[k], res.get(M.path[k], (None, 'not used', 0))[1], res.get(M.path[k], (None, '', 0))[2]) for k in used],
                      columns=['map', 'map_time', 'status', 'n_open'])
    st.to_csv(f'{DS}/pfss_log10fs_hourly_maps.csv', index=False)
    print(f'done in {(time.time() - t0) / 60:.1f} min; map status {st.status.str[:20].value_counts().to_dict()}; '
          f'hours without value {int(np.isnan(G[0]).sum())}; map age median {np.nanmedian(age):.2f} h, 99% {np.nanpercentile(age, 99):.1f} h', flush=True)
