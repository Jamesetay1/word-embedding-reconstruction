#!/usr/bin/bash
#SBATCH --job-name=RNNLM_charCNN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --output=logs/evaluation/%j.log

pwd; hostname; date
#module load python/3.10.8-gpu
#module load nvidia/cudnn/8.6.0-cuda-11.6

source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python evaluation.py \
--load './models/charCNN_500k_BiLSTM' \
--outf './output/charCNN_500k_BiLSTM.txt' \
--cuda

date