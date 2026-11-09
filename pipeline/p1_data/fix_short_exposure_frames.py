"""Replace AIA-193 frames taken in flare mode with very short exposures (EXPTIME < 0.1 s; quiet Sun at ~1 DN, so
dark-area features explode, e.g. 2017-09-06 12:00 at 5 ms) by the nearest synoptic frame within +-10 min with
QUALITY == 0 and a normal exposure (>= 1 s).  Steps:
  1. download replacements into $SWNN_SOLAR_DATA/SDO/AIA/LVL1.5/hourly/<year>/193 (as fetch_replacement_frames.py);
     manifest data/aia_short_exposure_replacements.csv
  2. append their rows to data/preprocessed_window/aia_fits_hourly_quality_eswf.csv (same per-frame code)
  3. pixel features: drop the short frames from data/pixel_features_aia.pkl, append the replacements
  4. NN1 dataset (data/nn1_dataset): recompute the affected hours from the replacement; if none was found, copy the
     previous hour's values (features, masks, dlon_center); update times.csv; remove the resampled caches.
Frames with EXPTIME < 0.1 s are excluded everywhere via aia_frames.usable() from now on.
"""
import concurrent.futures
import os
from pathlib import Path

import numpy as np
import pandas as pd

import folds
from fetch_replacement_frames import fetch, BASE, OUT_ROOT
from aia_frames import MIN_EXPTIME, NORMAL_EXPTIME

QCSV = 'data/preprocessed_window/aia_fits_hourly_quality_eswf.csv'
MANIFEST = 'data/aia_short_exposure_replacements.csv'
OFFSETS = [-2, 2, -4, 4, -6, 6, -8, 8, -10, 10]


def find_replacement(args):
    from astropy.io import fits
    hour, short_name = args
    tried = []
    for off in OFFSETS:
        ts = hour + pd.Timedelta(minutes=off)
        name = f'AIA{ts:%Y%m%d_%H%M}_0193.fits'
        if name == short_name:
            continue
        dst = OUT_ROOT / f'{ts:%Y}' / '193' / name
        part = dst.with_suffix('.fits.part') if not dst.exists() else None
        src = dst
        if part is not None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            got = fetch(f'{BASE}/{ts:%Y/%m/%d}/H{ts:%H}00/{name}', dst)
            if got is None:
                tried.append(f'{off:+d}:none'); continue
            src = got
        try:
            h = fits.getheader(src, 1); q, e = int(h.get('QUALITY', -1)), float(h.get('EXPTIME', 0))
        except Exception:
            q, e = -1, 0.0
        tried.append(f'{off:+d}:q{q:#x}:e{e:.3f}')
        if q == 0 and e >= NORMAL_EXPTIME:
            if src != dst:
                os.replace(src, dst)
            return hour, str(dst), off, 'ok', ' '.join(tried)
        if src != dst:
            Path(src).unlink(missing_ok=True)
    return hour, '', None, 'no normal-exposure frame within +-10 min', ' '.join(tried)


if __name__ == '__main__':
    q = pd.read_csv(QCSV, parse_dates=['obstime']); q['t'] = q.obstime.dt.round('h')
    short = q[(q.quality == 0) & (q.exptime < MIN_EXPTIME) & (q.t >= folds.DEV_START) & (q.t <= folds.DEV_END)]
    print(f'{len(short)} short-exposure frames (< {MIN_EXPTIME} s) in the development window', flush=True)
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(find_replacement, [(t, Path(p).name) for t, p in zip(short.t, short.path)]))
    man = pd.DataFrame(res, columns=['hour', 'replacement', 'offset_min', 'status', 'tried'])
    man['short_file'] = short.path.to_numpy(); man['short_exptime'] = short.exptime.to_numpy()
    man.to_csv(MANIFEST, index=False)
    print('replacements:', man.status.value_counts().to_dict(), flush=True)

    # 2. quality table
    from extract_fits_quality_eswf import _process
    from preprocess_solar_image import _get_aia_correction_table
    table = _get_aia_correction_table()
    new = sorted(set(man.loc[man.status == 'ok', 'replacement']) - set(q.path))
    with concurrent.futures.ProcessPoolExecutor(16) as ex:
        add = pd.DataFrame(list(ex.map(_process, [(p, table) for p in new])))
    q2 = pd.concat([q.drop(columns='t'), add], ignore_index=True); q2['obstime'] = pd.to_datetime(q2.obstime)
    q2.sort_values('obstime').to_csv(QCSV, index=False)
    print(f'quality table: +{len(add)} rows (QUALITY 0: {(add.quality == 0).sum()})', flush=True)

    # 3. pixel features
    import build_pixel_features as BP
    pf = pd.read_pickle('data/pixel_features_aia.pkl')
    n0 = len(pf); pf = pf[~pf.path.isin(set(short.path))]
    rows = [BP._run(('aia', p, table)) for p in new]
    rows = [r for r in rows if r.get('ok')]
    pf = pd.concat([pf, pd.DataFrame(rows).drop(columns=['ok'])], ignore_index=True)
    pf['obstime'] = pd.to_datetime(pf.obstime); pf = pf.sort_values('obstime').reset_index(drop=True)
    pf.to_pickle('data/pixel_features_aia.pkl')
    print(f'pixel features: {n0} -> {len(pf)} rows ({n0 - len(pf) + len(rows)} short removed, {len(rows)} replacements added)', flush=True)

    # 4. NN1 dataset
    import build_nn1_dataset as B
    D = Path('data/nn1_dataset')
    tt = pd.read_csv(D / 'times.csv', parse_dates=['time'])
    kidx = {t: k for k, t in enumerate(tt.time)}
    B._init(len(tt), table)
    F, ma, dc = B._G['F'], B._G['ma'], B._G['dc']
    for _, r in man.iterrows():
        k = kidx.get(pd.Timestamp(r.hour))
        if k is None:
            continue
        if r.status == 'ok':
            k_, st_a, _ = B._hour((k, r.replacement, None))          # AIA channels only; HMI unchanged
            assert st_a == 'ok', st_a
            tt.loc[k, 'aia_path'] = r.replacement
        else:                                                         # carry the previous hour forward
            F[:6, :, k] = F[:6, :, k - 1]; ma[:, k] = ma[:, k - 1]; dc[:, k] = dc[:, k - 1]
            tt.loc[k, 'aia_path'] = f'carried forward from {tt.time[k - 1]}'
    F.flush(); ma.flush(); dc.flush(); B._G['mh'].flush()
    tt.to_csv(D / 'times.csv', index=False)
    for c in D.glob('resampled_nphi*.npy'):
        c.unlink()
    print('NN1 dataset patched; resampled caches removed (rebuilt on next use)', flush=True)
