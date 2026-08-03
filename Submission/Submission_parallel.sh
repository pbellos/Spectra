#!/bin/bash
#SBATCH --job-name=Parallel.test.2
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:30:00

echo "Job started at $(date)"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29600
export WORLD_SIZE=$SLURM_NTASKS

echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"

srun --cpu-bind=none bash -c '
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
python /home/b5ao/pbellos.b5ao/Spectra/inv_predict_parallel.py --target bond_existence --dataset_type NMR_B --tag ParTrainFinal2_240 --predict No --debug False'

echo "Job finished at $(date)"

rm events.out.tfevents.*