#!/bin/bash

#SBATCH --job-name=james_train_n1-1_6-12-2023
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --output=kvq_restricted_experiment/results/logs/train_r%j.log

pwd; hostname; date
export HOME=/work/tc046/tc046/jamesetay1
module load pytorch/1.13.1-gpu
source /work/tc046/tc046/jamesetay1/subword-to-word/compact_reconstruction/venv/bin/activate

srun kvq_restricted_experiment/scripts/train_1-10_r.sh

date