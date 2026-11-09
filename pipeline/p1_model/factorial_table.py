"""Full table of the factorial test (2026-10-18): hourly metrics on the matched development test hours
(runs/sweep4/factorial_matched.csv, factorial_matched.py) joined with the PROSWIN high-speed-stream event metrics of
each variant (runs/dev_events_factorial/<cfg>/{metrics,events}.csv, from dev_events_series.py --runs runs/sweep4/<cfg>
--out runs/dev_events_factorial/<cfg> and dev_events_eval.py runs/dev_events_factorial/<cfg>). Event columns: hits,
misses, false alarms (187 observed streams), POD, FAR, TS, bias score, HSS-peak RMSE, median |timing error| and
median |peak-speed error| of the hits. The linear comparator and recurrence rows of the event part come from B0's
event run (same series as the paper).

  python pipeline/p1_model/factorial_table.py
Output: runs/sweep4/factorial_full.csv
"""
import pandas as pd

H = pd.read_csv('runs/sweep4/factorial_matched.csv').set_index('variant')
rows = {}
for cfg in ('B0', 'F_ball', 'F_lag', 'F_lin', 'F_lin_lag', 'G_nam'):
    d = f'runs/dev_events_factorial/{cfg}'
    M = pd.read_csv(f'{d}/metrics.csv').set_index('method')
    E = pd.read_csv(f'{d}/events.csv')
    for meth, key in (('NN', cfg), ('Linear', 'linear'), ('Recurrence', 'recurrence')):
        if key in rows:
            continue
        m, e = M.loc[meth], E[(E.method == meth) & (E.kind == 'hit')]
        rows[key] = dict(hits=int(m.tp), misses=int(m.fn), false_alarms=int(m.fp), pod=m.pod, far=m.far, ts=m.ts,
                         bias_score=m.bs, peak_rmse=m.peak_rmse, med_abs_dt_h=e.dt_h.abs().median(),
                         med_abs_dv=(e.pred_peak_v - e.obs_peak_v).abs().median(), omni_hours=int(m.n_hours))
R = H.join(pd.DataFrame(rows).T, how='left')
R.to_csv('runs/sweep4/factorial_full.csv')
pd.set_option('display.width', 250)
print(R.drop(columns=['label']).round(3).to_string())
