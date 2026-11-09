"""Part I case studies, stage 3 of 3: select the four case-study events from the B0 event list (runs/dev_events/events.csv,
dev_events_eval.py) by rules fixed before looking at the events.

  error of a hit   e = |Δt| / median|Δt| + |Δv| / median|Δv| over all B0 hits (Δt = predicted − observed peak time,
                   Δv = predicted − observed peak speed): timing and amplitude weighted equally
  1 best success   the hit with the smallest e
  2 typical        the hit with the median e
  3 failure        the missed stream with the highest observed peak speed
  4 interface      the hit with the steepest observed rise: the largest 24-h increase of the hourly OMNI speed within the
                   3 days before the observed peak
Eligibility (all cases): at least 70 % of the B0 matched ACE hours present within ±5 d of the observed peak, so each case
can be shown; the ±5-d windows of the cases do not overlap (peaks ≥ 10 d apart), cases chosen in the order listed.

  python pipeline/p1_model/dev_cases_select.py
Output: runs/dev_events/cases.csv
"""
import numpy as np
import pandas as pd

DIR = 'runs/dev_events'
COVER, HALF = 0.70, pd.Timedelta(days=5)

if __name__ == '__main__':
    E = pd.read_csv(f'{DIR}/events.csv', parse_dates=['obs_peak', 'pred_peak'])
    E = E[E.method == 'NN'].reset_index(drop=True)
    A = pd.read_csv(f'{DIR}/ace_axis.csv', parse_dates=['time']).set_index('time')
    S = pd.read_csv(f'{DIR}/series.csv', index_col=0, parse_dates=[0])
    cover = lambda t: A.index[(A.index >= t - HALF) & (A.index < t + HALF)].size / (2 * HALF / pd.Timedelta(hours=1))
    E['cover'] = [cover(t) if pd.notna(t) else np.nan for t in E.obs_peak]
    H = E[E.kind == 'hit'].copy()
    H['dv'] = H.pred_peak_v - H.obs_peak_v
    H['err'] = H.dt_h.abs() / H.dt_h.abs().median() + H.dv.abs() / H.dv.abs().median()

    def rise(t):
        o = S.obs[(S.index >= t - pd.Timedelta(days=3)) & (S.index <= t)].asfreq('1h')
        return float((o - o.shift(24)).max())
    H['rise24'] = [rise(t) for t in H.obs_peak]
    ok = H.cover >= COVER
    used, rows = [], []
    free = lambda t: all(abs(t - u) >= 2 * HALF for u in used)

    def take(case, rule, q):
        used.append(q.obs_peak)
        rows.append(dict(case=case, rule=rule, obs_peak=q.obs_peak, obs_peak_v=q.obs_peak_v,
                         pred_peak=q.get('pred_peak'), pred_peak_v=q.get('pred_peak_v'), dt_h=q.get('dt_h'),
                         dv=q.get('dv'), err=q.get('err'), rise24=q.get('rise24'), cover=q.cover))
    first = lambda X: X[[free(t) for t in X.obs_peak]].iloc[0]
    Hs = H[ok].sort_values('err')
    take('best', 'smallest combined error', first(Hs))
    med = Hs.err.median()
    take('typical', 'median combined error', first(Hs.iloc[(Hs.err - med).abs().argsort()]))
    Mi = E[(E.kind == 'miss') & (E.cover >= COVER)].sort_values('obs_peak_v', ascending=False)
    take('failure', 'largest missed stream', first(Mi))
    take('interface', 'steepest observed 24-h rise', first(H[ok].sort_values('rise24', ascending=False)))
    C = pd.DataFrame(rows)
    C.to_csv(f'{DIR}/cases.csv', index=False)
    pd.set_option('display.width', 200)
    print(f'B0 hits {len(H)} (eligible {ok.sum()}), median |dt| {H.dt_h.abs().median():.1f} h, median |dv| {H.dv.abs().median():.1f} km/s')
    print(C.round(2).to_string(index=False))
