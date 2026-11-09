"""Per hourly Level-1.5 synoptic FITS frame (AIA-193, 2011-2021): header QUALITY / EXPTIME /
T_OBS, and the pixel-level ESWF-style proxy computed exactly as compute_eswf_style_proxy.py
did on JP2 (on-disk 0.98 R, +/-7.5 deg central-meridian strip, % below 0.35 * on-disk median),
but from FITS DN/EXPTIME with degradation correction.
Output: data/preprocessed_window/aia_fits_hourly_quality_eswf.csv
"""
import concurrent.futures
import time

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from preprocess_solar_image import (apply_aia193_degradation_correction, _get_aia_correction_table,
                                    set_aia_correction_table)

HALF_WIDTH_DEG = 7.5
REL_THRESH = 0.35
OUT = 'data/preprocessed_window/aia_fits_hourly_quality_eswf.csv'


def _process(args):
    path, table = args
    try:
        set_aia_correction_table(table)
        with fits.open(path) as hdul:
            hdu = hdul[1] if len(hdul) > 1 else hdul[0]
            h = hdu.header
            data = np.asarray(hdu.data, dtype=np.float64)
        quality = int(h.get('QUALITY', -1))
        exptime = float(h['EXPTIME'])
        t_obs = Time(h.get('T_OBS', h.get('DATE-OBS')).rstrip('Z'))
        img = apply_aia193_degradation_correction(data / exptime, t_obs).astype(np.float64)
        cx, cy = float(h['CRPIX1']) - 1, float(h['CRPIX2']) - 1
        r_pix = float(h['RSUN_OBS']) / float(h['CDELT1'])
        ny, nx = img.shape
        yy, xx = np.ogrid[0:ny, 0:nx]
        on_disk = (((xx - cx) ** 2 + (yy - cy) ** 2) <= (r_pix * 0.98) ** 2) & np.isfinite(img)
        disk = img[on_disk]
        med = float(np.median(disk)) if disk.size else np.nan
        strip = on_disk & (np.abs(xx - cx) <= r_pix * np.sin(np.deg2rad(HALF_WIDTH_DEG)))
        sv = img[strip]
        pct = 100.0 * float(np.mean(sv < REL_THRESH * med)) if (sv.size > 50 and med > 0) else np.nan
        return dict(obstime=t_obs.datetime, path=path, quality=quality, exptime=exptime,
                    disk_median=med, eswf_pct=pct, n_strip=int(sv.size))
    except Exception as exc:
        return dict(obstime=pd.NaT, path=path, quality=-2, error=f'{type(exc).__name__}: {exc}')


def main():
    paths = pd.read_csv('data/ecliptic_map_cache/fits_hourly_paths.csv')['path'].tolist()
    table = _get_aia_correction_table()
    rows, t0 = [], time.time()
    with concurrent.futures.ProcessPoolExecutor(48) as ex:
        for i, r in enumerate(ex.map(_process, [(p, table) for p in paths], chunksize=16), 1):
            rows.append(r)
            if i % 10000 == 0:
                print(f'  {i}/{len(paths)} {time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows).sort_values('obstime').reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f'saved {OUT}: {len(df)} rows, {time.time()-t0:.0f}s', flush=True)
    print('QUALITY value counts (top 10):')
    print(df['quality'].map(lambda q: hex(q) if q >= 0 else q).value_counts().head(10).to_string())
    good = df['quality'] == 0
    print(f'QUALITY==0: {good.sum()}  ({good.mean()*100:.1f}%);  '
          f'invalid median among QUALITY==0: {(good & ~(df.disk_median > 0)).sum()};  '
          f'invalid median among QUALITY!=0: {(~good & ~(df.disk_median > 0)).sum()}')


if __name__ == '__main__':
    main()
