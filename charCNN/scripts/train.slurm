#!/usr/bin/bash
#SBATCH --job-name=CNN_500K
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --output=logs/training/%j.log
#SBATCH --account=tc046-jtaylor

pwd; hostname; date

source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python src/train.py \
--npy_path preprocessed/fastTextS_r.npy \
--model_name fastTextS_r*50 \
--batch_size 50 \
--filter_interval 50

date