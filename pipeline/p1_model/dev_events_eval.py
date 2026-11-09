"""Part I case studies, stage 2 of 3 (windnet env, like part2_events_eval.py): the PROSWIN high-speed-stream event
protocol on the development out-of-fold series (runs/dev_events/series.csv from dev_events_series.py).

For each method: their PredictionEvaluator with their HSSFinder (data/external/proswin_repo, unmodified; observed events
= the GFZ enhancement list, peaks of the prediction found by their detector, association window ±3 d), giving POD,
FAR, TS, bias score, timeline RMSE and HSS-peak RMSE, and the list of every event: hits (observed and predicted peak
date and speed, timing error = predicted − observed peak time), misses (observed peak) and false alarms (predicted peak).

  $PY_W pipeline/p1_model/dev_events_eval.py [runs/dev_events]
Output: runs/dev_events/metrics.csv, runs/dev_events/events.csv
"""
import logging
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'data/external/proswin_repo/src/src'))
from proswin.evaluation.hss_finder import HSSFinder  # noqa: E402
from proswin.evaluation.solar_wind_prediction_evaluator import PredictionEvaluator  # noqa: E402
from proswin.predictions import PointPrediction  # noqa: E402

logging.disable(logging.WARNING)
DIR = sys.argv[1] if len(sys.argv) > 1 else 'runs/dev_events'

if __name__ == '__main__':
    finder = HSSFinder(pathlib.Path('data/external/proswin_gfz'))
    D = pd.read_csv(f'{DIR}/series.csv', index_col=0, parse_dates=[0])
    rows, ev = [], []
    for meth in D.columns[1:]:
        E = PredictionEvaluator(PointPrediction(D[meth].rename('predicted_mean')), D.obs, finder, None)
        m = E.evaluate().drop('crps', errors='ignore')
        pp, op = E._get_hss_peaks()
        rows.append(dict(method=meth, n_hours=len(D), tp=int(pp.hit.sum()), fp=int(pp.false_alarm.sum()),
                         fn=int(op.miss.sum()), **m.to_dict()))
        for t, q in pp.iterrows():
            if q.hit:
                o = pd.Timestamp(q.associated)
                ev.append(dict(method=meth, kind='hit', obs_peak=o, obs_peak_v=float(op.loc[o, 'peak_value']), pred_peak=t,
                               pred_peak_v=float(D[meth].get(t, float('nan'))), dt_h=(t - o) / pd.Timedelta(hours=1)))
            elif q.false_alarm:
                ev.append(dict(method=meth, kind='false_alarm', pred_peak=t, pred_peak_v=float(D[meth].get(t, float('nan')))))
        for t, q in op.iterrows():
            if q.miss:
                ev.append(dict(method=meth, kind='miss', obs_peak=t, obs_peak_v=float(q.peak_value)))
    M, EV = pd.DataFrame(rows), pd.DataFrame(ev)
    M.to_csv(f'{DIR}/metrics.csv', index=False)
    EV.to_csv(f'{DIR}/events.csv', index=False)
    pd.set_option('display.width', 200)
    print(M.round(3).to_string(index=False))
    print(EV.groupby(['method', 'kind']).size().unstack())
