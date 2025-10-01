#!/bin/bash -l
#SBATCH -A m1820
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32

module load python
module load cudatoolkit
conda activate evolver

DB=/pscratch/sd/$USER/optuna/evolver.db
mkdir -p "$(dirname "$DB")"

# Launch 4 single-GPU workers. Each pins to one GPU via SLURM_LOCALID.
srun -u bash -lc '
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
python /global/homes/s/schenke/ML/jimwlk/tune_evolver_optuna.py \
  --data_root $SCRATCH/jimwlk_runs_32 \
  --out_root  $SCRATCH/evolver_tuning \
  --study_name evolver \
  --storage sqlite:///'"$DB"' \
  --n_trials 50 \
  --epochs 30 --prune
'  # total trials ≈ 4 * 50
