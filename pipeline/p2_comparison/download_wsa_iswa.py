"""Download the public CCMC iSWA WSA 5.4 solar-wind speed maps at 21.5 Rs (GONG_Z-driven), for the Part 2 comparison
(PART2_model_comparison.md). These are the standard WSA runs named in the data statement of Edward-Inatimi et al.
(2026). The archive is continuous only from 2022-12 on (checked 2026-10-13), so this covers 2022-12 .. 2025-07.

Source: https://iswa.gsfc.nasa.gov/iswa_data_tree/model/solar/WSA5.4/R21.5/WSA_VEL/GONG_Z/YYYY/MM/vel_*.fits
Output: $SWNN_SOLAR_DATA/WSA5.4_iSWA/GONG_Z/YYYY/MM/vel_*.fits
        data/external/wsa_iswa_manifest.csv (file, run time, size, status)

  python download_wsa_iswa.py [--start 2022-12] [--end 2025-07] [--workers 8]

Development period (Part I comparison, 2026-10-24): the public CCMC WSA 4.5 (GONG) maps, same format (coronal field
and speed at 21.5 Rs, 2° grid, the same speed equation), hourly 2006 .. 2020-11; Part I uses every map (61,712 files,
8.4 GB):
  python download_wsa_iswa.py --product wsa45 --start 2010-12 --end 2020-06
(--issues-from '<glob of forecast .npz>' fetches only the latest map at or before each issue time.)
"""
import argparse
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from data_paths import SOLAR

BASE = 'https://iswa.gsfc.nasa.gov/iswa_data_tree/model/solar/WSA5.4/R21.5/WSA_VEL/GONG_Z/'
OUT = str(SOLAR / 'WSA5.4_iSWA/GONG_Z')
MANIFEST = 'data/external/wsa_iswa_manifest.csv'
UA = {'User-Agent': 'Mozilla/5.0'}


def listdir(url):
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read().decode('utf8', 'ignore')
    return re.findall(r'href="([^"?/][^"]*)"', html)


def fetch(args):
    url, path = args
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, os.path.getsize(path), 'exists'
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
        with open(path + '.part', 'wb') as f:
            f.write(data)
        os.replace(path + '.part', path)
        return path, len(data), 'ok'
    except Exception as e:
        return path, 0, f'error: {type(e).__name__}: {e}'[:120]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2022-12')
    ap.add_argument('--end', default='2025-07')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--product', choices=['wsa54', 'wsa45'], default='wsa54')
    ap.add_argument('--issues-from', default='', help='glob of forecast .npz: fetch only the latest map <= each issue time')
    a = ap.parse_args()
    if a.product == 'wsa45':
        BASE = 'https://iswa.gsfc.nasa.gov/iswa_data_tree/model/solar/WSA4.5/WSA_VEL_GONG/'
        OUT = str(SOLAR / 'WSA4.5_iSWA/GONG')
        MANIFEST = 'data/external/wsa45_iswa_manifest.csv'
    jobs = []
    for m in pd.period_range(a.start, a.end, freq='M'):
        sub = f'{m.year}/{m.month:02d}/'
        try:
            names = [n for n in listdir(BASE + sub) if n.endswith('.fits')]
        except Exception as e:
            print(f'{sub}: listing failed ({e})', flush=True)
            continue
        jobs += [(BASE + sub + n, f'{OUT}/{sub}{n}') for n in names]
        print(f'{sub}: {len(names)} files', flush=True)
    if a.issues_from:
        import glob
        import numpy as np
        iss = np.unique(np.concatenate([pd.to_datetime(np.load(f, allow_pickle=True)['issue']).values
                                        for f in sorted(glob.glob(a.issues_from))]))
        rt = pd.to_datetime(pd.Series([j[0] for j in jobs]).str.extract(r'vel_(\d{12})')[0], format='%Y%m%d%H%M').values
        order = np.argsort(rt); rt = rt[order]; jobs = [jobs[i] for i in order]
        k = np.searchsorted(rt, iss, side='right') - 1
        keep = sorted(set(k[k >= 0].tolist()))
        jobs = [jobs[i] for i in keep]
        print(f'{len(iss)} issue times -> {len(jobs)} distinct latest maps', flush=True)
    print(f'{len(jobs)} files to fetch', flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(fetch, jobs))
    df = pd.DataFrame(res, columns=['path', 'bytes', 'status'])
    df['run_time'] = pd.to_datetime(df.path.str.extract(r'vel_(\d{12})')[0], format='%Y%m%d%H%M')
    df.sort_values('run_time').to_csv(MANIFEST, index=False)
    print(f'done: {df.status.str[:5].value_counts().to_dict()}, {df.bytes.sum() / 1e9:.2f} GB; manifest {MANIFEST}', flush=True)
