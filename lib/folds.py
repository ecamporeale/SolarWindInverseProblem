"""Single source of truth for the development split (train on solar cycle 24).

Development window: 2011-01-01T00:00 to 2020-06-01T12:00 UTC (last measured AIA-193
calibration epoch; also just after the cycle-24/25 minimum of Dec 2019), cut into 5
contiguous batches of equal clock-time length (~687.9 d). Batches are defined by dates,
not by counting observation rows, so data gaps cannot shift the boundaries.

Held-out test period: after DEV_END (cycle 25). Reserved; see PAPER_1.md.
"""
import numpy as np
import pandas as pd

DEV_START = pd.Timestamp('2011-01-01T00:00')
DEV_END = pd.Timestamp('2020-06-01T12:00')
N_BATCHES = 5
EDGES = pd.DatetimeIndex([DEV_START + (DEV_END - DEV_START) * k / N_BATCHES for k in range(N_BATCHES + 1)]).round('h')

# fold -> (train batches, val batch, test batch)
FOLDS = {1: ([0, 1, 2], 3, 4), 2: ([1, 2, 3], 4, 0), 3: ([2, 3, 4], 0, 1),
         4: ([3, 4, 0], 1, 2), 5: ([4, 0, 1], 2, 3)}


# Alternative split 'il20' (2026-10-15, experiment E1 in PAPER_3.md; after PROSWIN, Collin et al. 2026): the same
# development window cut into 20 equal chunks (~172 d), assigned round-robin: fold f (i = f - 1) validates on chunks
# j % 5 == i, tests on chunks j % 5 == (i + 1) % 5, trains on the rest. Training hours within IL20_GAP of any
# validation or test chunk are dropped (leakage through the 27-day recurrence; 54 d = two rotations), so every fold
# trains on all phases of the cycle. Each hour of the window is a test hour of exactly one fold.
# The default split 'b5' (FOLDS above) is unchanged.
SPLITS = ('b5', 'il20')
IL20_N = 20
IL20_GAP = pd.Timedelta(days=54)
IL20_EDGES = pd.DatetimeIndex([DEV_START + (DEV_END - DEV_START) * k / IL20_N for k in range(IL20_N + 1)]).round('h')
_HOUR = pd.Timedelta(hours=1)


def _edges(split):
    E = list(EDGES if split == 'b5' else IL20_EDGES)
    E[-1] = DEV_END + _HOUR                                          # last segment includes DEV_END, as batch_of
    return E


def batch_intervals(batches):
    """[(start, end)) of b5 batches."""
    E = _edges('b5')
    return [(E[k], E[k + 1]) for k in batches]


def as_intervals(x):
    """A list of b5 batch ids or a list of (start, end) intervals -> list of (start, end) intervals."""
    x = list(x)
    if x and not isinstance(x[0], (tuple, list)):
        return batch_intervals([int(k) for k in x])
    return [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in x]


def in_intervals(times, intervals) -> np.ndarray:
    t = pd.to_datetime(pd.Series(np.asarray(times)), format='mixed').to_numpy()
    m = np.zeros(len(t), bool)
    for s, e in as_intervals(intervals):
        m |= (t >= s.to_datetime64()) & (t < e.to_datetime64())
    return m


def fold_intervals(fold, split='b5'):
    """{'train', 'val', 'test'} -> list of [start, end) intervals (training already gap-trimmed for 'il20')."""
    if split == 'b5':
        tr, va, te = FOLDS[fold]
        return {'train': batch_intervals(tr), 'val': batch_intervals([va]), 'test': batch_intervals([te])}
    if split != 'il20':
        raise ValueError(f'unknown split {split!r}; known: {SPLITS}')
    i, E = fold - 1, _edges('il20')
    role = ['val' if j % 5 == i else 'test' if j % 5 == (i + 1) % 5 else 'train' for j in range(IL20_N)]
    out = {'train': [], 'val': [], 'test': []}
    j = 0
    while j < IL20_N:                                                # runs of consecutive chunks with the same role
        k = j
        while k + 1 < IL20_N and role[k + 1] == role[j]:
            k += 1
        s, e = E[j], E[k + 1]
        if role[j] == 'train':
            if j > 0:
                s = s + IL20_GAP
            if k < IL20_N - 1:
                e = e - IL20_GAP
        if e > s:
            out[role[j]].append((s, e))
        j = k + 1
    return out


def batch_of(times) -> np.ndarray:
    """Batch index 0..4 for each timestamp; -1 outside the development window.
    Batch k covers [EDGES[k], EDGES[k+1]); the last batch includes DEV_END."""
    t = pd.to_datetime(pd.Series(np.asarray(times)), format='mixed').to_numpy()
    b = np.searchsorted(EDGES.to_numpy(), t, side='right') - 1
    b = np.where(t == EDGES[-1].to_datetime64(), N_BATCHES - 1, b)
    return np.where((b >= 0) & (b < N_BATCHES), b, -1)


def is_heldout(times) -> np.ndarray:
    return pd.to_datetime(pd.Series(np.asarray(times)), format='mixed').to_numpy() > DEV_END.to_datetime64()


if __name__ == '__main__':
    for k in range(N_BATCHES):
        print(f'batch {k}: {EDGES[k]} -> {EDGES[k+1]}  ({(EDGES[k+1]-EDGES[k]).total_seconds()/86400:.1f} d)')
