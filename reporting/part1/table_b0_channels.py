"""Table of the frozen model B0's inputs per boundary cell (second review, item 27): the six selected AIA 193 Å features
(channels 0-5, plain-language definitions as in the appendix table of all candidates), their lagged copies, the PFSS
expansion factor and the back-mapped recurrence; 37 inputs in all.

  python reporting/part1/table_b0_channels.py
Output: reporting/tables/table_b0_channels.tex (label tab:channels)
"""
from pathlib import Path

from table_all_features import B0, definition

rows = [f'{ch} & {definition(f)} & 1\\\\' for f, ch in sorted(B0.items(), key=lambda kv: kv[1])]
rows += [r'6--33 & Channels 0--5 at $t-12$, $t-24$, $t-48$ and $t-72$~h, each lag with a flag marking a missing image & 28\\',
         r'34 & $\log_{10}$ of the PFSS expansion factor $f_s$ at the cell (hourly GONG map) & 1\\',
         r'35--36 & Back-mapped recurrence: mean ACE speed traced back to this longitude one rotation earlier, and its validity flag & 2\\']
tex = r'''\begin{table}
\caption{Inputs of the frozen model B0 for one boundary cell (Carrington longitude, hour $t$). Channels 0--5 are the six AIA 193~\AA\ features selected in Section~\ref{sec:screening}, rescaled to their central-meridian equivalent; the full definitions and the screening statistics of all candidates are in Table~\ref{tab:allfeatures}. All inputs are standardized with training-batch statistics.}
\label{tab:channels}\centering\small
\begin{tabular}{lp{0.72\linewidth}r}\hline
Channel & Input & Values\\\hline
''' + '\n'.join(rows) + r'''
\hline
 & Total & 37\\\hline
\end{tabular}
\end{table}
'''
out = Path('reporting/tables/table_b0_channels.tex')
out.write_text(tex)
print(f'wrote {out}')
