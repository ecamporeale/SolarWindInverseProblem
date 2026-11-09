# Source from the repository root before running anything:   source env.sh
# All scripts are run from the repository root (data paths are relative to it). Every code folder is on PYTHONPATH,
# so modules import each other by name.
export SWNN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_dirs="lib pipeline/p1_data pipeline/p1_features pipeline/p1_model pipeline/p2_heldout pipeline/p2_comparison reporting/part1 checks"
_pp=""; for _d in $_dirs; do _pp="$_pp:$SWNN_ROOT/$_d"; done
export PYTHONPATH="${_pp#:}"
unset _dirs _pp _d
# Python interpreters of the three environments (specs in environments/); override before sourcing if needed.
export PY_L="${PY_L:-python}"        # LiveWind: everything not listed below
export PY_H="${PY_H:-python}"        # huxt: part1_wsa_profiles.py, part1_wsa_huxt_test.py (HUXt 5.0.3)
export PY_W="${PY_W:-python}"        # windnet: dev_events_eval.py (PROSWIN evaluation code)
# Large external data (not in the repository); see README.md and lib/data_paths.py.
# export SWNN_SOLAR_DATA=/path/to/solar      # SDO/AIA/LVL1.5/hourly, SDO/HMI/fits, GONG_MAG, WSA4.5_iSWA/GONG
# export SWNN_ACE_DATA=/path/to/ace_1h       # monthly ACE SWEPAM 1-h files
