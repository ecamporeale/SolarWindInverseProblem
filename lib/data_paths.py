"""Locations of the large external data sets (not in the repository). Set them before running the data steps:

  export SWNN_SOLAR_DATA=/path/to/solar      # SDO/AIA/LVL1.5/{hourly,all}, SDO/AIA/4096_jp2, SDO/HMI/fits, GONG_MAG, WSA4.5_iSWA
  export SWNN_ACE_DATA=/path/to/ace_1h       # monthly ACE SWEPAM 1-h files (preprocess_ACE.py)

Defaults: data/solar and data/ace_1h under the repository root (env.sh of the working copy sets this machine's paths).
"""
import os
from pathlib import Path

SOLAR = Path(os.environ.get('SWNN_SOLAR_DATA', 'data/solar'))
ACE_1H = Path(os.environ.get('SWNN_ACE_DATA', 'data/ace_1h'))
