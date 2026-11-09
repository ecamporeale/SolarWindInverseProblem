"""Replacement AIA-193 synoptic frames for development-window hours without a usable frame.

The hourly download took the HH:00 frame (or the nearest within +-10 min). When that frame was
flagged (QUALITY != 0; e.g. the daily 21:00 UTC frames flagged 0x10000) the hour was lost, although
the synoptic archive has a frame every 2 minutes. For every such hour this tries offsets
+-2, +-4, ..., +-10 min (skipping the frame already rejected) and keeps the first frame with
QUALITY == 0; flagged candidates are deleted.
Writes into the same folder as the hourly download; manifest: data/aia_synoptic_replacements_manifest.csv
"""
import concurrent.futures
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from astropy.io import fits
from data_paths import SOLAR

BASE = 'http://jsoc.stanford.edu/data/aia/synoptic'
OUT_ROOT = SOLAR / 'SDO/AIA/LVL1.5/hourly'
OFFSETS = [-2, 2, -4, 4, -6, 6, -8, 8, -10, 10, 0]
MANIFEST = 'data/aia_synoptic_replacements_manifest.csv'


def fetch(url, dst):
    part = dst.with_suffix('.fits.part')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(part, 'wb') as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if part.stat().st_size < 100_000:
                part.unlink(missing_ok=True)
                return None
            return part
        except urllib.error.HTTPError as e:
            part.unlink(missing_ok=True)
            if e.code == 404:
                return None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            part.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
    return None


def do_hour(args):
    hour, rejected = args
    tried = []
    for off in OFFSETS:
        ts = hour + pd.Timedelta(minutes=off)
        name = f'AIA{ts:%Y%m%d_%H%M}_0193.fits'
        if name in rejected:
            continue
        dst = OUT_ROOT / f'{ts:%Y}' / '193' / name
        if dst.exists():                      # already on disk (e.g. from a previous run)
            q = int(fits.getheader(dst, 1).get('QUALITY', -1))
            tried.append(f'{off:+d}:exists:{q:#x}')
            if q == 0:
                return hour, str(dst), off, 'ok', ' '.join(tried)
            continue
        part = fetch(f'{BASE}/{ts:%Y/%m/%d}/H{ts:%H}00/{name}', dst)
        if part is None:
            tried.append(f'{off:+d}:none')
            continue
        try:
            q = int(fits.getheader(part, 1).get('QUALITY', -1))
        except Exception:
            q = -1
        tried.append(f'{off:+d}:{q:#x}')
        if q == 0:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part, dst)
            return hour, str(dst), off, 'ok', ' '.join(tried)
        part.unlink(missing_ok=True)
    return hour, '', '', 'no clean frame within +-10 min', ' '.join(tried)


def main():
    tt = pd.read_csv('data/nn1_dataset/times.csv', parse_dates=['time'])
    bad = tt[tt.aia_status != 'ok']
    q = pd.read_csv('data/preprocessed_window/aia_fits_hourly_quality_eswf.csv', parse_dates=['obstime'])
    q['t'] = q.obstime.dt.round('h')
    rej = q[q.quality != 0].groupby('t').path.apply(lambda s: {Path(p).name for p in s}).to_dict()
    tasks = [(h, rej.get(h, set())) for h in bad.time]
    print(f'{len(tasks)} hours to repair', flush=True)
    rows, t0 = [], time.time()
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(do_hour, tasks), 1):
            rows.append(r)
            if i % 500 == 0 or i == len(tasks):
                ok = sum(x[3] == 'ok' for x in rows)
                print(f'  {i}/{len(tasks)} repaired {ok} ({(time.time()-t0)/60:.1f} min)', flush=True)
    m = pd.DataFrame(rows, columns=['hour', 'file', 'offset_min', 'status', 'tried'])
    m.to_csv(MANIFEST, index=False)
    print('done', m.status.value_counts().to_dict(), flush=True)


if __name__ == '__main__':
    main()
