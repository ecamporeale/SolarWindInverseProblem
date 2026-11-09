"""PFSS expansion factor on the NN1 longitude grid, hourly, for the development window (NN1 input channel).

Per day d: PFSS (sunkit-magex, rss 2.5, nrho 30) from the FIRST GONG mrzqs map of day d (causal for all hours of d);
field lines traced down from the source surface at the 128 model longitudes (j * 360/128 deg, the periodic grid)
and Earth's heliographic latitude B0 at 12 UT of d. log10 f_s of each open line; lines that are not open (or fail)
take the value of the nearest open longitude of the same day. Every hour of day d gets the day's values.
Days without a GONG map: previous day's values (forward fill).
Output: data/nn1_dataset/pfss_log10fs_nphi128.npy  float32 [128, Nt_hourly], aligned with times.csv;
        data/nn1_dataset/pfss_log10fs_days.csv  (day, map, n_open, status)
"""
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
from data_paths import SOLAR

warnings.filterwarnings('ignore')
GONG = str(SOLAR / 'GONG_MAG')
NPHI, RSS, NRHO = 128, 2.5, 30
LONS = np.arange(NPHI) * 360.0 / NPHI


def day_fs(day):
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import sun
    import sunkit_magex.pfss as pfss
    files = sorted(glob.glob(f'{GONG}/{day:%Y}/{day:%Y%m%d}/mrzqs*.fits.gz'))
    if not files:
        return day, None, '', 0, 'no_map'
    try:
        m = sunpy.map.Map(files[0]).resample([360, 180] * u.pix)
        pout = pfss.pfss(pfss.Input(m, NRHO, RSS))
        b0 = float(sun.B0(day + pd.Timedelta(hours=12)).to_value(u.deg))
        s = SkyCoord(LONS * u.deg, np.full(NPHI, b0) * u.deg, (RSS - 0.01) * u.R_sun, frame=pout.coordinate_frame)
        fls = pfss.tracing.PerformanceTracer().trace(s, pout)
        ef = np.asarray(fls.expansion_factors, float)
        ok = np.array([fl.is_open for fl in fls]) & np.isfinite(ef) & (ef > 0)
        out = np.full(NPHI, np.nan)
        out[ok] = np.log10(ef[ok])
        if ok.sum() == 0:
            return day, None, os.path.basename(files[0]), 0, 'no_open'
        idx = np.flatnonzero(ok)
        for j in np.flatnonzero(~ok):                         # nearest open longitude (circular)
            d = np.abs((idx - j + NPHI // 2) % NPHI - NPHI // 2)
            out[j] = out[idx[d.argmin()]]
        return day, out.astype(np.float32), os.path.basename(files[0]), int(ok.sum()), 'ok'
    except Exception as e:
        return day, None, os.path.basename(files[0]), 0, f'error: {type(e).__name__}: {e}'[:160]


if __name__ == '__main__':
    from multiprocessing import Pool
    times = pd.to_datetime(pd.read_csv('data/nn1_dataset/times.csv', usecols=['time'])['time'])
    days = pd.date_range(times.iloc[0].floor('D'), times.iloc[-1].floor('D'), freq='D')
    t = time.time()
    with Pool(48) as pool:
        res = pool.map(day_fs, list(days), chunksize=4)
    res.sort(key=lambda r: r[0])
    tab = pd.DataFrame([dict(day=r[0], map=r[2], n_open=r[3], status=r[4]) for r in res])
    vals = pd.DataFrame([r[1] if r[1] is not None else np.full(NPHI, np.nan, np.float32) for r in res], index=days).ffill().bfill()
    G = vals.reindex(times.dt.floor('D')).to_numpy().T.astype(np.float32)          # [128, Nt]
    np.save('data/nn1_dataset/pfss_log10fs_nphi128.npy', G)
    tab.to_csv('data/nn1_dataset/pfss_log10fs_days.csv', index=False)
    print(f'{len(days)} days in {time.time() - t:.0f} s; status: {tab.status.str[:30].value_counts().to_dict()}; '
          f'open lines per day median {tab.n_open.median():.0f}/128; log10 fs p5/p50/p95 = '
          f'{np.nanpercentile(G, 5):.2f}/{np.nanpercentile(G, 50):.2f}/{np.nanpercentile(G, 95):.2f}; shape {G.shape}')
