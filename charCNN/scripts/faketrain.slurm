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

module load pytorch/1.13.1-gpu

python src/train.py \
--npy_path preprocessed/10k.npy \
--model_name temp \
--batch_size 50 \
--filter_interval 1

date