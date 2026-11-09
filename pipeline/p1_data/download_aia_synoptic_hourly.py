"""Download an hourly subset of the JSOC AIA synoptic archive (Level 1.5, 1024x1024,
4x4-binned, raw DN -- same product as $SWNN_SOLAR_DATA/SDO/AIA/LVL1.5/all, verified
bit-identical) for AIA-193, as the FITS replacement for the Helioviewer JP2 dataset.

Source:  http://jsoc.stanford.edu/data/aia/synoptic/YYYY/MM/DD/HHH00/AIAYYYYMMDD_HHMM_0193.fits
Target:  $SWNN_SOLAR_DATA/SDO/AIA/LVL1.5/hourly/YYYY/193/<same filename>

One frame per hour: HH:00, else the nearest 2-min neighbour within +-10 min.
Resumable (existing files skipped), atomic writes (.part then rename), manifest CSV of
what was fetched / missing so gaps are explicit rather than silent.

Usage: python download_aia_synoptic_hourly.py START END [--workers 8]
       e.g. 2011-01-01 2022-01-01  (END exclusive)
"""
import argparse
import concurrent.futures
import csv
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from data_paths import SOLAR

BASE = 'http://jsoc.stanford.edu/data/aia/synoptic'
OUT_ROOT = SOLAR / 'SDO/AIA/LVL1.5/hourly'
WAVE = '0193'
MIN_BYTES = 100_000
OFFSETS_MIN = [0, 2, -2, 4, -4, 6, -6, 8, -8, 10, -10]


def url_for(ts):
    return (f"{BASE}/{ts:%Y/%m/%d}/H{ts:%H}00/AIA{ts:%Y%m%d_%H%M}_{WAVE}.fits",
            f"AIA{ts:%Y%m%d_%H%M}_{WAVE}.fits")


def fetch(url, dst, retries=4):
    """Returns 'ok', 'missing' (404) or raises after retries."""
    part = dst.with_suffix('.fits.part')
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(part, 'wb') as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if part.stat().st_size < MIN_BYTES:
                part.unlink(missing_ok=True)
                return 'missing'
            os.replace(part, dst)
            return 'ok'
        except urllib.error.HTTPError as e:
            part.unlink(missing_ok=True)
            if e.code == 404:
                return 'missing'
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            part.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f'failed after {retries} retries: {url}')


def do_hour(hour):
    out_dir = OUT_ROOT / f'{hour:%Y}' / '193'
    out_dir.mkdir(parents=True, exist_ok=True)
    for off in OFFSETS_MIN:
        ts = hour + pd.Timedelta(minutes=off)
        url, name = url_for(ts)
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size >= MIN_BYTES:
            return (hour, name, off, 'exists')
        try:
            status = fetch(url, dst)
        except RuntimeError as e:
            return (hour, name, off, f'error: {e}')
        if status == 'ok':
            return (hour, name, off, 'ok')
    return (hour, '', '', 'missing')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('start')
    ap.add_argument('end')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--manifest', default='data/aia_synoptic_hourly_manifest.csv')
    a = ap.parse_args()

    hours = pd.date_range(a.start, a.end, freq='1h', inclusive='left')
    print(f'{len(hours)} hours {hours[0]} -> {hours[-1]}, workers={a.workers}', flush=True)
    new_manifest = not Path(a.manifest).exists()
    counts, nbytes, t0 = {}, 0, time.time()
    with open(a.manifest, 'a', newline='') as mf:
        w = csv.writer(mf)
        if new_manifest:
            w.writerow(['hour', 'file', 'offset_min', 'status'])
        with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
            for i, (hour, name, off, status) in enumerate(ex.map(do_hour, hours), 1):
                w.writerow([hour.isoformat(), name, off, status])
                key = status.split(':')[0]
                counts[key] = counts.get(key, 0) + 1
                if status == 'ok':
                    nbytes += (OUT_ROOT / f'{hour:%Y}' / '193' / name).stat().st_size
                if i % 500 == 0 or i == len(hours):
                    el = time.time() - t0
                    mf.flush()
                    print(f'  {i}/{len(hours)} {counts} {nbytes/1e9:.2f} GB new, '
                          f'{nbytes/1e6/max(el,1e-9):.1f} MB/s, elapsed {el/60:.1f} min, '
                          f'eta {(len(hours)-i)*el/i/60:.0f} min', flush=True)
    print('done', counts, flush=True)


if __name__ == '__main__':
    main()
