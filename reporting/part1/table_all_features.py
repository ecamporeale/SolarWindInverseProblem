"""Appendix table of the Part I manuscript: all candidate features screened, in ranking order (Figure 3a), with
plain-language definitions, mean within-batch r and 95% rotation-block interval, and their use in B0 / the linear
comparator. Reads data/final_feature_ranking_pixel.csv (pipeline/p1_features/final_feature_ranking.py).

  python reporting/part1/table_all_features.py
Output: reporting/tables/table_all_features.tex (LaTeX table, label tab:allfeatures)
"""
import re
from pathlib import Path

import pandas as pd

B0 = {'A: area f=0.30, w=20, off=0': 0, 'B: ecliptic band |lat|<=15': 1, 'A: area f=0.35, w=12.5, off=0': 2,
      'D: mean log(I/med) of dark pixels, band |lat|<=30': 3, 'B: hole-boundary distance, off=+10': 4,
      'A: ESWF definition (f=0.35, w=7.5, off=0)': 5}
LIN = {'H: weak-field fraction <10 G, w=7.5, |lat|<=30', 'H: mean |B_r|, w=7.5, |lat|<=30'}


def side(o):
    return f'{abs(o)}$^\\circ$ {"east" if o < 0 else "west"} of the meridian'


def definition(f):
    m = re.match(r'A: area f=([\d.]+), w=([\d.]+), off=(-?\d+)', f)
    if m:
        o = int(m.group(3))
        return (f'Dark area ($f={m.group(1)}$), strip of half-width ${m.group(2)}^\\circ$ '
                + ('centred on the meridian' if o == 0 else 'centred ' + side(o)))
    if f.startswith('A: ESWF'):
        return r'Dark area ($f=0.35$), half-width $7.5^\circ$ (ESWF coronal-hole area)'
    m = re.match(r'B: ecliptic band \|lat\|<=(\d+)', f)
    if m:
        return f'Dark fraction within $|\\beta|\\leq{m.group(1)}^\\circ$, central strip'
    m = re.match(r'B: hole-boundary distance, off=([+-]?\d+)', f)
    if m:
        o = int(m.group(1))
        return ('Signed distance from the ecliptic to the nearest hole boundary, '
                + ('at the meridian' if o == 0 else side(o)) + ' (positive inside a hole)')
    if f.startswith('B: reach'):
        return r'Smallest $|\beta|$ reached by a coronal hole in the central strip'
    if f.startswith('C: AIA dark-fraction'):
        return r'Dark fraction in $5^\circ$ latitude bins of the central strip (profile, ridge)'
    if f.startswith('C: AIA log'):
        return r'Mean $\log_{10}(I/I_{\mathrm{med}})$ in $5^\circ$ latitude bins of the central strip (profile, ridge)'
    m = re.match(r'D: (.*), (band \|lat\|<=30|strip)$', f)
    if m:
        reg = r'central strip, $|\beta|\leq30^\circ$' if m.group(2).startswith('band') else r'central strip, $|\beta|\leq70^\circ$'
        q = {'integrated darkness': r'Integrated darkness $\sum\max(0,\log_{10}0.35-\log_{10}I/I_{\mathrm{med}})/N$',
             'soft-threshold area': r'Dark area with a logistic (soft) threshold at $0.35\,I_{\mathrm{med}}$',
             'mean log(I/med) of dark pixels': r'Mean $\log_{10}(I/I_{\mathrm{med}})$ of the dark pixels',
             '5th pct log(I/med)': r'5th percentile of $\log_{10}(I/I_{\mathrm{med}})$',
             '10th pct log(I/med)': r'10th percentile of $\log_{10}(I/I_{\mathrm{med}})$'}[m.group(1)]
        return f'{q}, {reg}'
    m = re.match(r'H: PIL distance, sigma=(\d+), off=(-?\d+)', f)
    if m:
        o = int(m.group(2))
        return (f'Distance from the ecliptic to the polarity inversion line of $B_r$ smoothed at ${m.group(1)}^\\circ$'
                + ('' if o == 0 else ', ' + side(o)))
    m = re.match(r'H: flux imbalance, w=20, \|lat\|<=(\d+)', f)
    if m:
        return f'Flux imbalance $|\\sum B_r|/\\sum|B_r|$, half-width $20^\\circ$, $|\\beta|\\leq{m.group(1)}^\\circ$'
    fixed = {'H: |net B_r|': r'Net flux $|\sum B_r|$, half-width $20^\circ$, $|\beta|\leq70^\circ$',
             'H: weak-field': r'Fraction of pixels with $|B_r|<10$~G, half-width $7.5^\circ$, $|\beta|\leq30^\circ$',
             'H: mean |B_r|': r'Mean $|B_r|$, half-width $7.5^\circ$, $|\beta|\leq30^\circ$',
             'H: HMI latitude': r'$|B_r|$, net flux and imbalance in latitude bins (profiles, ridge)',
             'H: low-order SH (l<=2) PIL': r'Inversion-line distance of the $l\leq2$ harmonic fit to the visible field',
             'H: low-order SH |B|': r'$|B|$ of the $l\leq2$ harmonic fit at the sub-Earth point',
             'H: low-order SH dipole': r'Dipole tilt of the $l\leq2$ harmonic fit',
             'OLD NN1 input: AIA': r'Mean AIA intensity, $|\beta|\leq3^\circ$, central strip (input of an earlier prototype)',
             'OLD NN1 input: HMI': r'Mean HMI $|B_{\mathrm{los}}|$, $|\beta|\leq3^\circ$, central strip (input of an earlier prototype)',
             'REF: 27-day': r'Reference: ACE speed one solar rotation (27~d) before the target hour',
             'REF: persistence': r'Reference: ACE speed at the image time (persistence)'}
    for k, v in fixed.items():
        if f.startswith(k):
            return v
    raise ValueError(f'no definition for {f!r}')


def num(x):
    s = f'{x:+.2f}'
    return '0.00' if s in ('+0.00', '-0.00') else s


if __name__ == '__main__':
    R = pd.read_csv('data/final_feature_ranking_pixel.csv')
    rows = []
    for _, q in R.iterrows():
        use = f'B0 ch{B0[q.feature]}' if q.feature in B0 else ('Lin.' if q.feature in LIN else '')
        fam = 'P' if q.feature.startswith('OLD') else 'ref.' if q.feature.startswith('REF') else q.feature[0]
        rows.append(f'{int(q["rank"])} & {fam} & {definition(q.feature)} & ${num(q.mean_r)}$ [${num(q.lo)}$, ${num(q.hi)}$] & {use}\\\\')
    tex = r'''\begin{table}[p]
    \caption{All candidate features screened (Section~\ref{sec:screening}), in the order of Figure~\ref{fig:ranking}a. Families: A, dark area in a full-height strip; B, proximity of dark regions to the ecliptic; C, latitude profiles; D, intensity within the dark regions; H, HMI magnetic field; P, inputs of an earlier prototype; ref., reference predictors. Unless stated otherwise, ``dark'' means $I<0.35\,I_{\mathrm{med}}$, the central strip is $|\Delta\mathrm{lon}|\leq7.5^\circ$, and full-height strips span $|\beta|\leq70^\circ$; coronal holes are connected dark regions of at least 5~deg$^2$. $r$: Pearson correlation with the ACE speed four days after the image, mean of the five within-batch values, with the 95\% rotation-block bootstrap interval. Use: image channel 0--5 of the frozen model B0 (the six selected AIA features), or Lin.\ for the two HMI features used, with the six AIA features, only by the linear comparator.}
    \label{tab:allfeatures}\centering\footnotesize
    \begin{tabular}{rlp{0.58\linewidth}cc}\hline
     & Fam. & Definition & $r$ [95\% interval] & Use\\\hline
    ''' + '\n'.join(rows) + r'''
    \hline
    \end{tabular}
    \end{table}
    '''
    out = Path('reporting/tables/table_all_features.tex')
    out.write_text(tex)
    print(f'wrote {out} ({len(rows)} features)')
