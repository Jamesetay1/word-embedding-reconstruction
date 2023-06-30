#!/usr/bin/bash
#SBATCH --job-name=CNN_500K
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --output=logs/training/%j.log

pwd; hostname; date
#module load python/3.10.8-gpu
#module load nvidia/cudnn/8.6.0-cuda-11.6

source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python src/train.py \
--npy_path data/500k.npy \
--model_name 500k_50c \
--batch_size 50 \
--filter_interval 50

date