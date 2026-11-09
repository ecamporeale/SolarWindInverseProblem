"""Shared style for the Part I figures (figure_*.py): palette, matplotlib defaults, save().

Palette: categorical slots of the dataviz reference palette, fixed per method across all figures.
Every figure is written as png (200 dpi) and pdf to figures/part1/.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = Path('figures/part1')
INK, INK2, GRID, MUTED = '#0b0b0b', '#52514e', '#e4e3df', '#8a8985'
C = {'nn': '#2a78d6', 'nn2': '#eb6834', 'persistence': '#1baf7a', 'recurrence': '#eda100', 'linear': '#e87ba4',
     'eswf': '#008300', 'climatology': '#6250d6', 'reference': '#8a8985'}
LBL = {'nn': 'Reference model', 'linear': 'Linear, 6 AIA + 2 HMI', 'recurrence': '27-day recurrence',
       'climatology': 'Climatology', 'mse': 'NN, MSE loss'}
SURF_IN, SURF_MODEL, SURF_OUT = '#eef3fb', '#fdf0ea', '#eef7f2'
BLUES = LinearSegmentedColormap.from_list('blues', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95',
                                                    '#0d366b'])

plt.rcParams.update({'font.size': 11.5, 'axes.labelsize': 12, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
                     'legend.fontsize': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                     'axes.titlesize': 12, 'axes.titlelocation': 'left'})


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, ha='right', va='bottom', fontsize=13, weight='bold', color=INK)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT / name}.png/.pdf', flush=True)
