#!/usr/bin/bash
#SBATCH --job-name=RNNLM_random
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --output=logs/training/%j.log

pwd; hostname; date
#module load python/3.10.8-gpu
#module load nvidia/cudnn/8.6.0-cuda-11.6

source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python train.py \
--glove '../data/fastText_embeddings/random500k.vec' \
--rnn_type 'LSTM' \
--direction 'both' \
--wo_char \
--tok_emb 300 \
--cuda \
--save './models/random_500k_BiLSTM'

date

