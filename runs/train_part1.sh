#!/bin/bash
# Train the Part I configurations (runs/part1_configs.txt), 5 folds x seeds 42 and 43 each, one model at a time.
#   bash runs/train_part1.sh [config ...]          (default: all 14; GPU=<index> selects the GPU, default 0)
# The released checkpoints are kept: a model whose best.pt exists is skipped unless FORCE=1.
# Each model needs ~20-45 GB of GPU memory and several hours (A100); the paper ran them in parallel on 8 GPUs.
cd "$(dirname "$0")/.." && source env.sh
mkdir -p tmp
COMMON="--nn1-arch feature --spacecraft ACE --chunks-per-batch 8 --batch-overlap-hours 150 --batch-group-size 8 --disable-nn2 --loss-weighting combined --lr 2e-3 --nn1-fill train-median --nn1-lag-hours 12,24,48,72 --freeze-accel --nn1-vmax 800 --no-speed-interp --epochs 300 --patience 50 --plot-every 50"
want=" $* "
while IFS='|' read -r id folder extra; do
  [[ -z "$id" || "$id" == \#* ]] && continue
  [[ $# -gt 0 && "$want" != *" $id "* ]] && continue
  for seed in 42 43; do for fold in 1 2 3 4 5; do
    d=$folder/$id/seed$seed/fold$fold
    if [ -f $d/best.pt ] && [ "${FORCE:-0}" != 1 ]; then echo "skip $d (best.pt exists; FORCE=1 retrains)"; continue; fi
    mkdir -p $d; echo "$(date '+%F %T') train $d"
    TMPDIR=$PWD/tmp $PY_L -u lib/train_joint_forecast.py --fold $fold --seed $seed $COMMON $extra \
      --cuda-index ${GPU:-0} --plot-dir $d/plots --checkpoint $d/best.pt --output-model $d/final.pt > $d/train.log 2>&1 \
      || echo "FAILED $d (see $d/train.log)"
  done; done
done < runs/part1_configs.txt
