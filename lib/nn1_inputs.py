"""NN1 inputs from the per-longitude dataset (data/nn1_dataset, PAPER_2.md) and date-defined batches
from folds.py, for train_joint_forecast.py (--nn1-dataset / --fold).

image_slice channels [C=11, Nphi, Nt] on the model time grid (dt = config.dt):
  0-7  set-2a features (PAPER_2.md section 3), rescaled to their central-meridian equivalent
       (offset_rescaling, fitted on the training batches); standardized inside the NN1 model
  8    dlon_center / 60 (column offset from central meridian), 0 where not valid
  9    AIA valid (1/0): NN1 outputs a speed only here; elsewhere a gradient-free constant
  10   HMI available (1/0): channels 6-7 are 0 where this is 0
  11+   temporal context (config.nn1_lag_hours): per lag L, channels 0-7 at hour t - L (rescaled with that
       frame's own offset), its AIA-valid and HMI flags (10 channels per lag), read from the full hourly
       dataset so chunk starts have their history too
  then  extra channels (config.nn1_extra_channels, EXTRA_REGISTRY), one per name, e.g. 'pfss_log10fs' = daily PFSS
       expansion factor on the model longitudes (pfss_fs_grid.py); not rescaled, standardized in NN1
Batches: batch k spans [EDGES[k] - overlap_hours, EDGES[k+1]) (last batch to DEV_END inclusive;
batch 0 starts at DEV_START). The first overlap_hours of every batch are warm-up and are not scored
(train_joint_forecast._transient_skip_steps); warm-up uses the preceding batch's images, never its targets.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import folds

DATA = Path('data/nn1_dataset')
N_FEAT = 8
CH_DLON, CH_VALID, CH_HMI = 8, 9, 10
N_CH = 11
EXTRA_REGISTRY = {'pfss_log10fs': 'pfss_log10fs_nphi128.npy',                    # name -> [Nphi, Nt_hourly] file in DATA
                  'pfss_log10fs_hourly': 'pfss_log10fs_hourly_nphi128.npy',      # most recent GONG map <= hour
                  'bm_recurrence': 'bm_recurrence_nphi128.npy',                  # back-mapped ACE speed, previous passage
                  'bm_recurrence_valid': 'bm_recurrence_valid_nphi128.npy'}      # 1/0 flag for the above
EXTRA_NAMES = list(EXTRA_REGISTRY)


def load_extra(name, nphi):
    a = np.load(DATA / EXTRA_REGISTRY[name], mmap_mode='r')
    if a.shape[0] != nphi:
        raise ValueError(f'{name}: {a.shape[0]} longitudes, model has {nphi}')
    return a


def _interp_matrix(n_src, n_dst):
    """Periodic linear interpolation weights [n_dst, n_src] from 1-deg columns to n_dst points."""
    W = np.zeros((n_dst, n_src), np.float64)
    pos = np.arange(n_dst) * n_src / n_dst
    i0 = np.floor(pos).astype(int) % n_src
    fr = pos - np.floor(pos)
    W[np.arange(n_dst), i0] += 1 - fr
    W[np.arange(n_dst), (i0 + 1) % n_src] += fr
    return W


def load_resampled(nphi):
    """[N_CH, nphi, Nt_hourly] float32 + hourly times; cached after the first call."""
    cache = DATA / f'resampled_nphi{nphi}.npy'
    times = pd.to_datetime(pd.read_csv(DATA / 'times.csv', usecols=['time'])['time'])
    if cache.exists():
        return np.load(cache, mmap_mode='r'), times
    F = np.load(DATA / 'features.npy', mmap_mode='r')
    ma = np.load(DATA / 'mask_aia.npy').astype(np.float64)
    mh = np.load(DATA / 'mask_hmi.npy').astype(np.float64)
    dc = np.load(DATA / 'dlon_center.npy').astype(np.float64)
    W = _interp_matrix(F.shape[1], nphi)
    den_a, den_h = W @ ma, W @ mh
    va, vh = den_a > 0.5, den_h > 0.5
    out = np.zeros((N_CH, nphi, F.shape[2]), np.float32)
    for c in range(N_FEAT):
        m, den, v = (ma, den_a, va) if c < 6 else (mh, den_h, vh)
        num = W @ (np.asarray(F[c], np.float64) * m)
        out[c] = np.where(v & (va if c >= 6 else True), num / np.maximum(den, 1e-9), 0.0)
    num = W @ (np.nan_to_num(dc) * ma)
    out[CH_DLON] = np.where(va, num / np.maximum(den_a, 1e-9) / 60.0, 0.0)
    out[CH_VALID] = va
    out[CH_HMI] = va & vh
    out[6:8] *= out[CH_HMI]
    np.save(cache, out)
    print(f'NN1 inputs resampled 360 -> {nphi} longitudes, cached at {cache}', flush=True)
    return np.load(cache, mmap_mode='r'), times


def slice_to_model_grid(X, times, start, nt, dt_seconds):
    """Nearest hourly frame for every model step (within 30 min); zeros (= not valid) outside the dataset."""
    t0 = times.iloc[0]
    tgt = pd.date_range(start, periods=nt, freq=pd.Timedelta(seconds=dt_seconds))
    k = np.rint((tgt - t0) / pd.Timedelta(hours=1)).astype(int)
    ok = (k >= 0) & (k < X.shape[2])
    out = np.zeros((X.shape[0], X.shape[1], nt), np.float32)
    out[:, :, ok] = X[:, :, k[ok]]
    return out


RESC_EDGES = np.arange(-60.0, 60.01, 5.0)
RESC_MID = 0.5 * (RESC_EDGES[:-1] + RESC_EDGES[1:])
MULTIPLICATIVE = ()                      # all channels affine: E[CM value | value at offset] (zeros from hidden holes need an intercept)
REF_HALF = 2.0                            # meridian reference: mean over |dlon_c| <= 2 deg


def offset_rescaling(train_batches):
    """Per channel and 5-deg dlon_c bin, regression of a longitude's central-meridian value on its value at
    offset dlon_c within the same disk passage, using passages in the training batches only (b5 batch ids or
    [start, end) intervals, folds.as_intervals).
    Returns alpha, beta [8, n_bins]: CM-equivalent = alpha + beta * value (alpha = 0 for area channels)."""
    F = np.load(DATA / 'features.npy', mmap_mode='r')
    MA, MH = np.load(DATA / 'mask_aia.npy', mmap_mode='r'), np.load(DATA / 'mask_hmi.npy', mmap_mode='r')
    DC = np.load(DATA / 'dlon_center.npy', mmap_mode='r')
    times = pd.to_datetime(pd.read_csv(DATA / 'times.csv', usecols=['time'])['time'])
    in_train = folds.in_intervals(times.to_numpy(), train_batches)
    nb = len(RESC_MID)
    S = {k: np.zeros((N_FEAT, nb)) for k in ('n', 'x', 'y', 'xx', 'xy')}   # x = value at offset, y = CM value
    for j in range(F.shape[1]):
        dc = np.asarray(DC[j], np.float64)
        idx = np.flatnonzero(np.isfinite(dc) & in_train)
        if not idx.size:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 24) + 1)
        for c in range(N_FEAT):
            f = np.asarray(F[c, j], np.float64)
            m = np.asarray((MA if c < 6 else MH)[j], bool)
            for run in runs:
                run = run[m[run]]
                ref = run[np.abs(dc[run]) <= REF_HALF]
                if run.size < 24 or not ref.size:
                    continue
                b = np.digitize(dc[run], RESC_EDGES) - 1
                ok = (b >= 0) & (b < nb)
                b, x = b[ok], f[run][ok]
                y = np.full(x.shape, f[ref].mean())
                for k, v in (('n', np.ones_like(x)), ('x', x), ('y', y), ('xx', x * x), ('xy', x * y)):
                    S[k][c] += np.bincount(b, weights=v, minlength=nb)
    alpha, beta = np.zeros((N_FEAT, nb)), np.ones((N_FEAT, nb))
    for c in range(N_FEAT):
        n, sx, sy, sxx, sxy = (S[k][c] for k in ('n', 'x', 'y', 'xx', 'xy'))
        if c in MULTIPLICATIVE:
            beta[c] = np.where(sxx > 0, sxy / np.maximum(sxx, 1e-12), 1.0)
        else:
            vx = sxx / n - (sx / n) ** 2
            beta[c] = (sxy / n - sx * sy / n ** 2) / np.maximum(vx, 1e-12)
            alpha[c] = sy / n - beta[c] * sx / n
    return alpha, beta


def apply_rescaling(img, alpha, beta):
    """In place on an image_slice [N_CH, Nphi, Nt]: every valid feature cell -> its CM-equivalent value."""
    d = np.clip(img[CH_DLON] * 60.0, RESC_MID[0], RESC_MID[-1])
    pos = np.interp(d, RESC_MID, np.arange(len(RESC_MID)))
    i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, len(RESC_MID) - 1); w = pos - i0
    for c in range(N_FEAT):
        a = alpha[c, i0] * (1 - w) + alpha[c, i1] * w
        b = beta[c, i0] * (1 - w) + beta[c, i1] * w
        valid = img[CH_VALID if c < 6 else CH_HMI] > 0.5
        img[c] = np.where(valid, a + b * img[c], 0.0)
    return img


def feature_stats(X, times, batches, alpha=None, beta=None):
    """Mean/std of channels 0-7 over valid cells in the given (training) batches or intervals, after rescaling if given."""
    sel = np.where(folds.in_intervals(times.to_numpy(), batches))[0][::3]
    Xs = np.array(X[:, :, sel], dtype=np.float32)
    if alpha is not None:
        apply_rescaling(Xs, alpha, beta)
    mean, std = np.zeros(N_FEAT), np.ones(N_FEAT)
    for c in range(N_FEAT):
        v = Xs[c][Xs[CH_VALID if c < 6 else CH_HMI] > 0.5]
        mean[c], std[c] = float(v.mean()), float(max(v.std(), 1e-6))
    return mean, std


def image_slice(X, times, EX, start, nt, dt_seconds, alpha=None, beta=None, lags=()):
    """NN1 input [11 + 10 * len(lags) + len(EX), Nphi, nt] from `start` on a dt_seconds grid: current channels,
    then per lag L the 8 features + AIA-valid + HMI flags at t - L, then the extra channels (not rescaled)."""
    img = slice_to_model_grid(X, times, start, nt, dt_seconds)
    if alpha is not None:
        apply_rescaling(img, alpha, beta)
    blocks = [img]
    for L in lags:
        lg = slice_to_model_grid(X, times, start - pd.Timedelta(hours=L), nt, dt_seconds)
        if alpha is not None:
            apply_rescaling(lg, alpha, beta)
        blocks.append(np.concatenate([lg[:N_FEAT], lg[CH_VALID:CH_VALID + 1], lg[CH_HMI:CH_HMI + 1]], axis=0))
    for ex in EX:
        blocks.append(slice_to_model_grid(ex, times, start, nt, dt_seconds))
    return np.concatenate(blocks, axis=0) if len(blocks) > 1 else img


def build_fold_batch_specs(config, v_obs_raw_df, train_batches, chunks_per_batch=1, rescale=True, roles=None,
                           box_days=None):
    """Date-defined batch specs: each development batch split into `chunks_per_batch` contiguous chunks.

    Chunk j of batch k scores [e_j, e_{j+1}) (the e_j tile [EDGES[k], EDGES[k+1]) exactly) and starts
    overlap_hours earlier as warm-up, taken from the preceding images (never scored). Spec 'batch_idx'
    is a running chunk id; 'dev_batch' is the development batch it belongs to (its fold role).
    Sets config.Nt (longest chunk) and the training-batch feature statistics.

    roles (folds.fold_intervals: {'train', 'val', 'test'} -> [start, end) intervals): instead of the five b5
    batches, chunk every interval of every role; 'dev_batch' is then the role name and train_batches is ignored
    (training intervals = roles['train']). box_days: chunk count per interval = ceil(length / box_days)
    (equal chunks, none longer than box_days) instead of chunks_per_batch.
    """
    from inverse_solver import _build_obs_tracks_for_batch
    X, times = load_resampled(int(config.Nphi))
    overlap = pd.Timedelta(hours=float(getattr(config, 'overlap_hours', 0.0)))
    hour = pd.Timedelta(hours=1)
    if roles is None:
        segs = [(k, s, e) for k, (s, e) in enumerate(folds.batch_intervals(range(folds.N_BATCHES)))]
    else:
        train_batches = roles['train']
        segs = sorted(((r, s, e) for r, iv in roles.items() for s, e in iv), key=lambda x: x[1])
    windows = []
    for k, lo, hi in segs:
        n = chunks_per_batch if box_days is None else int(np.ceil((hi - lo) / pd.Timedelta(days=float(box_days))))
        edges = [lo + ((hi - lo) * j / n).round('h') for j in range(n)] + [hi]
        for j in range(n):
            windows.append((k, max(edges[j] - overlap, folds.DEV_START), edges[j + 1]))
    max_hours = max(int((e - s) / hour) for _, s, e in windows)
    config.Nt = int(np.ceil(max_hours * 3600.0 / config.dt))
    config.t_end = config.Nt * config.dt
    alpha = beta = None
    if rescale:
        alpha, beta = offset_rescaling(train_batches)
        config.nn1_rescale_alpha, config.nn1_rescale_beta = alpha.tolist(), beta.tolist()
        print(f'Off-meridian rescaling fitted on batches {train_batches} (CM-equivalent values)', flush=True)
    mean, std = feature_stats(X, times, train_batches, alpha, beta)
    config.nn1_feature_mean, config.nn1_feature_std = mean.tolist(), std.tolist()
    print(f'Fold batches from folds.py: ' + (f'{chunks_per_batch} chunk(s) per development batch' if box_days is None
          else f'{len(windows)} chunks of <= {box_days} d over {len(segs)} role intervals') + f', Nt={config.Nt} '
          f'({config.Nt * config.dt / 3600:.0f} h model window), warm-up {overlap / hour:.0f} h; '
          f'feature stats from batches {train_batches}', flush=True)
    extras = list(getattr(config, 'nn1_extra_channels', None) or ())
    EX = [load_extra(nm, int(config.Nphi))[None] for nm in extras]                # each [1, Nphi, Nt_h]
    if extras:
        tr = folds.in_intervals(times.to_numpy(), train_batches)
        config.nn1_extra_mean = [float(np.nanmean(ex[0][:, tr])) for ex in EX]
        config.nn1_extra_std = [float(max(np.nanstd(ex[0][:, tr]), 1e-6)) for ex in EX]
        print(f'Extra NN1 channels {extras}: training mean {config.nn1_extra_mean}, std {config.nn1_extra_std}', flush=True)
    lags = [float(h) for h in (getattr(config, 'nn1_lag_hours', None) or ())]
    if lags:
        print(f'Temporal context: features at t - {lags} h ({10 * len(lags)} extra channels)', flush=True)
    specs = []
    for cid, (k, s, e) in enumerate(windows):
        tracks = _build_obs_tracks_for_batch(v_obs_raw_df, s, e, config, 'cpu')
        hours = int((e - s) / hour)
        specs.append({'batch_idx': cid, 'dev_batch': k, 'batch_start_time': str(s), 'batch_end_time': str(e),
                      'speed_hourly': np.full(hours, np.nan), 'obs_tracks': tracks,
                      'image_slice': image_slice(X, times, EX, s, config.Nt, config.dt, alpha, beta, lags)})
        n_obs = int(np.isfinite(tracks[0]['v_obs'].numpy()).sum()) if tracks else 0
        print(f'  chunk {cid:2d} ({"batch " if roles is None else ""}{k}): {s} -> {e}  ({hours} h; valid input cells '
              f'{specs[-1]["image_slice"][CH_VALID].mean()*100:.1f}%; {n_obs} usable hourly obs)', flush=True)
    return specs
