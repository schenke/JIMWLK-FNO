#!/bin/bash -l
#SBATCH -A m1820
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16        # give both ranks some CPU headroom
#SBATCH -L scratch
#SBATCH -J 4gpu
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

module load pytorch
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

python - <<'PY'
import torch; torch.set_float32_matmul_precision('high')
print("torch", torch.__version__, "| cuda?", torch.cuda.is_available(), "| ndev:", torch.cuda.device_count())
PY

srun --cpu-bind=cores \
torchrun --nproc_per_node=4 --standalone \
  /global/u2/s/schenke/ML/jimwlk/train_evolver_cuda_opt.py \
    --data_root "$SCRATCH/jimwlk_runs_256/backup" \
    --out "$SCRATCH/outputs_evolver" \
    --rollout_k 2  --rollout_consistency 0.05   --semigroup_prob 0.25   --semigroup_weight 0.05\
    --batch 16 --accum 2 --lr 1e-3 --min_lr 5e-5\
    --width 24 --blocks 8 --modes 48\
    --qs_weight 0 --dipole_weight 1\
    --trace_weight 0 --quad_weight 0.4\
    --loops_weight 0 --geo_weight 0.\
    --workers 4 --y_gain 1\
    --E1 0 --E2 20 --epochs 30\
    --nll_param_compose --nll_fullcov --nll_compose brownian\
    --nll_weight 0.05 --bch_weight 0 --energy_weight 0 --energy_stride 4\
    --channels_last --amp\
    --moment_weight 0.1 --scheduler cosine\
    --warmup_epochs 2 --warmup_start_factor 0.5\
    --y_map linear

