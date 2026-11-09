"""OMNI time base for the Part 2 comparison (PART2_model_comparison.md §2.1): move hourly series labelled on our ACE time
axis (label HH:00 = mean of HH:00..HH:59) to OMNI's bow-shock-nose time axis, the way OMNI does it.

Each hourly value is spread over its 60 minutes; every minute t is moved to t + (X_ACE(t) - X_BSN) / v(t), with X_ACE
ACE's GSE X position (data/ace_heldout/ACE_GSE_position.txt, daily, interpolated) and X_BSN the bow-shock nose distance
(fixed, BSN_RE Earth radii); the minutes are then averaged into hourly OMNI bins (label HH:00 = mean of HH:00..HH:59).
For a FORECAST, v is the forecast speed itself (no observation used, so the shift is causal); for the observed ACE
series, v is the observed speed (used only to validate the procedure against OMNI).

Also: load_omni() = OMNI hourly speed (PROSWIN data publication, data/external/proswin_gfz/solar_wind_speed.csv,
OMNIWeb hourly, bow-shock time-shifted) with the same ICME intervals as our ACE target removed (HELIO4CAST ICMECAT v2.3,
L1 icme_start .. mo_end, via preprocess_time_window).
"""
import numpy as np
import pandas as pd

RE_KM = 6371.0
BSN_RE = 13.0                                   # typical bow-shock nose distance (Farris & Russell model, quiet conditions)
POS_FILE = 'data/ace_heldout/ACE_GSE_position.txt'
OMNI_FILE = 'data/external/proswin_gfz/solar_wind_speed.csv'


def ace_x_km(index):
    p = pd.read_csv(POS_FILE, sep=r'\s+')
    t = pd.to_datetime(p.Year.astype(str), format='%Y') + pd.to_timedelta(p.DOY - 1, unit='D') + pd.to_timedelta(p.Secofday, unit='s')
    x = pd.Series(p['GSE_X(km)'].to_numpy(float), index=t).sort_index()
    x = x[~x.index.duplicated()]
    return np.interp(index.asi8, x.index.asi8, x.to_numpy())


def to_omni_time(values, speed=None, bsn_re=BSN_RE):
    """values: hourly pd.Series on the ACE label axis; speed: hourly pd.Series used for the delay (default: values).
    Returns an hourly pd.Series on the OMNI label axis."""
    v = values.dropna()
    sp = (values if speed is None else speed).reindex(v.index)
    ok = np.isfinite(sp.to_numpy()) & (sp.to_numpy() > 100)
    v, sp = v[ok], sp[ok]
    mins = np.arange(60)
    t_min = v.index.asi8[:, None] + mins[None, :] * 60_000_000_000            # ns
    dx = ace_x_km(pd.DatetimeIndex(v.index + pd.Timedelta(minutes=30))) - bsn_re * RE_KM
    delay_ns = (dx / sp.to_numpy())[:, None] * 1e9
    arr = pd.to_datetime((t_min + delay_ns).ravel().astype(np.int64))
    s = pd.Series(np.repeat(v.to_numpy(), 60), index=arr)
    return s.groupby(s.index.floor('h')).mean().sort_index()


RC_FILE = 'data/external/rc_icmetable2.htm'


def load_rc_icmes():
    """Richardson & Cane near-Earth ICME list (Caltech ACE ASC icmetable2, fetched 2026-10-14): one row per event with
    disturbance time, ICME start, ICME end (UT)."""
    import re
    h = open(RC_FILE, encoding='latin-1').read()
    out = []
    for r in re.findall(r'<tr[^>]*>(.*?)</tr>', h, re.S | re.I):
        c = [re.sub(r'<[^>]+>', '', x).strip() for x in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, re.S | re.I)]
        d0 = re.match(r'(\d{4}/\d{2}/\d{2} \d{4})', c[0]) if len(c) >= 3 else None
        d2 = re.match(r'(\d{4}/\d{2}/\d{2} \d{4})', c[2]) if len(c) >= 3 else None
        if d0 and d2:                                                    # cells may carry annotations, e.g. '(A)'
            out.append((pd.to_datetime(d0.group(1), format='%Y/%m/%d %H%M'), pd.to_datetime(d2.group(1), format='%Y/%m/%d %H%M')))
    return pd.DataFrame(out, columns=['disturbance', 'icme_end'])


def rc_mask(index):
    """True where an hour lies between an R&C disturbance time and the ICME end."""
    L = load_rc_icmes()
    bad = np.zeros(len(index), bool)
    for s, e in zip(L.disturbance, L.icme_end):
        bad |= (index >= s.floor('h')) & (index <= e.ceil('h'))
    return bad


def load_omni(mask='icmecat'):
    """OMNI hourly speed; mask = 'icmecat' (same ICME intervals as our ACE target), 'rc' (Richardson & Cane: disturbance ..
    ICME end) or 'none'."""
    import preprocess_time_window as P
    o = pd.read_csv(OMNI_FILE, index_col=0, parse_dates=True)['solar_wind_speed'].astype(float)
    o = o[(o > 0) & (o < 3000)]
    if mask == 'none':
        return o
    if mask == 'rc':
        return o[~rc_mask(o.index)]
    cme = P.load_cme_catalog(P.DEFAULT_CME_CATALOG_PATH, ['ACE'], o.index.min().to_pydatetime(), o.index.max().to_pydatetime(), verbose=False)
    keep = np.ones(len(o), bool)
    for s, e in zip(pd.to_datetime(cme.icme_start_time), pd.to_datetime(cme.mo_end_time)):
        keep &= ~((o.index >= s.floor('h')) & (o.index <= e.ceil('h')))
    return o[keep]


if __name__ == '__main__':
    held = pd.read_csv('data/preprocessed_window/multi_spacecraft_obs_heldout_20200401_20250731.csv', parse_dates=['Time'])
    ace = held[held.Spacecraft_ID == 'ACE'].set_index('Time')['Speed'].sort_index().resample('1h').mean()
    omni = load_omni()
    ace = ace[ace.index >= '2020-06-01 12:00']
    for nm, s in (('no shift', ace), ('constant +1 h', ace.shift(1, freq='h')), ('speed-dependent (OMNI-like)', to_omni_time(ace))):
        j = s.dropna().index.intersection(omni.index)
        d = s[j] - omni[j]
        print(f'ACE -> OMNI, {nm:28s}: n {len(j)}, RMS diff {np.sqrt((d ** 2).mean()):.2f}, mean {d.mean():+.2f}, r {np.corrcoef(s[j], omni[j])[0, 1]:.4f}')
