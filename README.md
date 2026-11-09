# Learned inner boundary for the ambient solar wind, Part I: code

Code and trained models for

> E. Camporeale, A. Hu, P. Mayank, T. E. Berger, *Forecasting the Ambient Solar Wind With a Physics-Informed Learned Inner Boundary: Part I—Differentiable Inverse Reconstruction From Solar Images* (submitted).

The repository contains only what reproduces the paper's results: 90 Python files (the scripts behind every figure,
table and number, and the library modules they import), a launcher for training, the 140 trained models of the
paper (14 configurations × 5 folds × 2 seeds) and the environment specifications. Data and intermediate results are
not included. They are either public (sources below) or produced by the scripts.

## Layout

| folder | contents |
|---|---|
| `lib/` | differentiable solver (`forward_model.py`, `inverse_solver.py`), boundary networks (`boundary.py`), training (`train_joint_forecast.py`), folds, inputs, look-alike search (`knn_core.py`), data paths (`data_paths.py`) |
| `pipeline/p1_data/` | download and preprocessing of ACE, AIA, HMI and GONG data; features; the model's input dataset |
| `pipeline/p1_features/` | feature ranking and backward elimination (Section 4) |
| `pipeline/p1_model/` | look-alike and noise analysis (Section 5); evaluation, ablations, factorial test, interpretable models, events and cases, WSA comparison (Section 7) |
| `pipeline/p2_heldout/`, `pipeline/p2_comparison/` | three shared modules: the block bootstrap of score differences (`heldout_scores.py`, with `p1_model/realtime_wcut_test.py`) and the WSA archive download (`download_wsa_iswa.py`) |
| `reporting/part1/` | the manuscript's figures (`figures/fig_*.py`) and generated tables |
| `runs/` | trained models (`runs/sweep4/<config>/seed<s>/fold<k>/best.pt`, `runs/sweep5_lasso/...`), `train_part1.sh`, `part1_configs.txt` |
| `checks/` | regression check of the evaluation |
| `environments/` | conda and pip specifications |

Run everything from the repository root after `source env.sh`. All data paths are relative to the root, except the
large solar data sets, whose location is set with `SWNN_SOLAR_DATA` and `SWNN_ACE_DATA` (`lib/data_paths.py`).

## Environments

Three Python environments, specified in `environments/`:

- **LiveWind** (`LiveWind.conda.yml`, Python 3.10, PyTorch with CUDA, SunPy, aiapy, sunkit-magex). Runs everything
  not listed below.
- **huxt** (`huxt.pip.txt`, HUXt 5.0.3). Runs the WSA profiles and WSA + HUXt.
- **windnet** (`windnet.pip.txt`). Runs `dev_events_eval.py`, which uses the PROSWIN evaluation code.

Point `PY_L`, `PY_H` and `PY_W` at their interpreters before `source env.sh` (the default is `python`).

The event evaluation uses the PROSWIN code of Collin et al. (2026), unmodified:

```
git clone https://github.com/danielcollin96/proswin data/external/proswin_repo/src
```

## External data

| data | source | used by |
|---|---|---|
| ACE SWEPAM hourly speed | https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/ | `fetch_spacecraft_data.py` |
| ACE position | https://izw1.caltech.edu/ACE/ASC/DATA/pos_att/ACE_GSE_position.txt | `fetch_spacecraft_data.py` |
| SDO/AIA 193 Å synoptic level 1.5, hourly (≈ 96,000 frames) | http://jsoc.stanford.edu/data/aia/synoptic | `download_aia_synoptic_hourly.py` |
| SDO/HMI magnetograms (two linear-comparator features) | JSOC `hmi.M_720s` | `build_pixel_features.py --instrument hmi` |
| GONG synoptic magnetograms (`mrzqs`) | https://gong.nso.edu/data/magmap/ | `pfss_fs_grid_hourly.py`, `pfss_connectivity.py` |
| WSA 4.5 (GONG) speed maps at 21.5 R☉ | https://iswa.gsfc.nasa.gov/iswa_data_tree/model/solar/WSA4.5/WSA_VEL_GONG/ | `download_wsa_iswa.py --product wsa45` |
| HELIO4CAST ICMECAT v2.3 | https://helioforecast.space/icmecat → `data/HELIO4CAST_ICMECAT_v23.csv` | ICME masking, figures |
| Richardson & Cane ICME list | https://izw1.caltech.edu/ACE/ASC/DATA/level3/icmetable2.htm → `data/external/rc_icmetable2.htm` | ICME masking |
| PROSWIN high-speed-stream data | https://doi.org/10.5880/GFZ.OJSJ.2026.001 → `data/external/proswin_gfz/` | event evaluation |

## Reproducing the paper

The steps run in this order. Cost: m = minutes, h = hours, D = days or terabyte downloads. "GPU" marks steps that
need one (≥ 40 GB for training).

### 1. Data (Section 3; D)

```
python pipeline/p1_data/fetch_spacecraft_data.py --start 2011-01-01 --end 2021-12-31 --spacecraft ACE
python pipeline/p1_data/merge_observations_2011_2021.py         # ICME-masked hourly ACE target
python pipeline/p1_data/download_aia_synoptic_hourly.py 2011-01-01 2021-12-31
python pipeline/p1_data/extract_fits_quality_eswf.py             # frame quality and usability
python pipeline/p1_data/fetch_replacement_frames.py              # replacements for hours without a usable frame
python pipeline/p1_data/extract_replacement_quality.py
python pipeline/p1_data/fix_short_exposure_frames.py             # flare-mode frames with very short exposures
python pipeline/p1_data/build_pixel_features.py --instrument aia  # the 42 candidate features (also --instrument hmi)
python pipeline/p1_data/build_nn1_dataset.py                     # inputs per Carrington longitude and hour
python pipeline/p1_data/pfss_fs_grid_hourly.py                   # PFSS expansion factor, hourly GONG (pfss_fs_grid.py: daily)
python pipeline/p1_data/bm_recurrence_grid.py                    # back-mapped 27-day recurrence
```

Each script's docstring gives its inputs, outputs and options.

### 2. Feature ranking and selection (Section 4; h)

```
FEATURE_SOURCE=pixel python pipeline/p1_features/final_feature_ranking.py
FEATURE_SOURCE=pixel python pipeline/p1_features/backward_elimination.py
python pipeline/p1_features/logistic_elimination.py
python pipeline/p1_features/ranking_alignment_scan.py            # choice of the 96-h lag
python pipeline/p1_features/test_feature_offset_mapping.py       # centre-to-limb correlations (Supplement S2)
python reporting/part1/table_all_features.py                     # table of all candidates
```

### 3. Input ambiguity and noise (Section 5; h, GPU)

```
KNN_ALIGN_BETA=0 python pipeline/p1_model/pfss_connectivity.py --workers 48
python pipeline/p1_model/knn_event_check.py                      # Table tab:lookalikes (fixed 96-h alignment)
python pipeline/p1_model/knn_other_batch_check.py
python pipeline/p1_model/overnight_t1_timing.py
python reporting/part1/figures/fig_degeneracy.py
NOISE_SOURCE=traced python pipeline/p1_model/noise_decomposition.py    # Gamma test and Table tab:budget (needs B0, step 4)
```

### 4. Training (Section 6; h per model, GPU)

The trained models are included. To retrain:

```
GPU=0 FORCE=1 bash runs/train_part1.sh            # all 14 configurations, 140 models
GPU=0 FORCE=1 bash runs/train_part1.sh B0 B0_mse  # selected configurations
```

`runs/part1_configs.txt` lists the configurations. B0 is the reference model; B0_mse uses the MSE loss; A1–A4 and
S_64-32-16 are the ablations; F_ball, F_lag, F_lin and F_lin_lag are the link and linear-map variants; G_nam is the
NAM; Q_lin and Q_nam0.3 are the four-coefficient and four-curve models. The choice of the acceleration parameters is
tested by re-scoring a trained model over a grid of values (no retraining):
`python pipeline/p1_model/profile_accel.py --run runs/sweep4/B0/seed42 --fold 1`.

### 5. Evaluation (Section 7; h, GPU)

Out-of-fold predictions and matched baselines, for every configuration and seed:

```
for c in runs/sweep4/{B0,B0_mse,A1_no_bm,A2_daily_fs,A3_with_hmi,A4_no_fs,S_64-32-16,F_ball,F_lag,F_lin,F_lin_lag,G_nam} \
         runs/sweep5_lasso/{Q_lin,Q_nam0.3}; do
  for s in 42 43; do python pipeline/p1_model/eval_matched_baselines.py --run $c/seed$s; done
done
python pipeline/p1_model/summarize_sweep.py runs/sweep4
```

Then:

| paper item | command |
|---|---|
| Table tab:performance | `b0_matched_S.py`, `b0_skill_metrics.py` |
| Table tab:skill | `b0_skill_metrics.py`, `b0_skill_bootstrap.py`, `part1_wsa_compare.py` |
| Table tab:factorial, Figure fig:condrmse_links | `factorial_matched.py`, `factorial_table.py`, `interface_check.py`, `fig_condrmse.py --set factorial` |
| NAM pairs (Section 6) | `nam_pair_selection.py` |
| Table tab:coefficients, Figure fig:qnam | `lasso_refit_scores.py`, `fig_qnam.py` |
| Table tab:ablations | `ablation_stats.py runs/sweep4` |
| Table tab:channels | `reporting/part1/table_b0_channels.py` |
| Table tab:events, Figure fig:cases | `dev_events_series.py`, `$PY_W dev_events_eval.py`, `dev_cases_select.py`, `fig_cases.py` |
| Figure fig:condrmse | `fig_condrmse.py` |
| Figures fig:pipeline, fig:features, fig:magnetic_inputs, fig:reconstruction, fig:hss | `fig_pipeline.py`, `fig_aia_features.py`, `fig_magnetic_inputs.py`, `fig_reconstruction.py`, `fig_loss.py` |

The scripts are in `pipeline/p1_model/` and `reporting/part1/figures/`. Figures are written to `figures/part1/`.

### 6. WSA 4.5 comparison (Section 7; h)

```
python pipeline/p2_comparison/download_wsa_iswa.py --product wsa45 --start 2010-12 --end 2020-06   # 8.4 GB
$PY_H pipeline/p1_model/part1_wsa_profiles.py --workers 32
$PY_H pipeline/p1_model/part1_wsa_huxt_test.py --workers 40          # WSA + HUXt on B0's test chunks
python pipeline/p1_model/part1_wsa_solver_test.py                  # WSA boundary + this paper's solver
python pipeline/p1_model/part1_wsa_ballistic_test.py
python pipeline/p1_model/part1_wsa_compare.py                      # scores, bootstrap, bias-aware measures
python pipeline/p1_model/dev_events_series.py --with-wsa --out runs/dev_events_wsa && $PY_W pipeline/p1_model/dev_events_eval.py runs/dev_events_wsa
```

## Reproducibility notes

- Training on GPUs is not bit-for-bit deterministic, so retrained models differ slightly from the released ones.
  The paper's scores are those of the released checkpoints.
- `python checks/check_b5_regression.py <gpu>` checks that the released B0 (seed 42, fold 1) reproduces its stored test
  predictions (to 3e-5 km/s). It needs the input dataset of step 1.
- All scores are cross-validated over the five development batches (2011–2020). The independent cycle-25 test is in Part II.

## Licence and citation

See `LICENSE` and `CITATION.cff`.
