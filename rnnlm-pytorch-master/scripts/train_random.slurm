#!/usr/bin/bash
#SBATCH --job-name=RNNLM_random
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --output=logs/training/%j.log

pwd; hostname; date
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python train.py \
--glove '../data/random_embeddings/random500k.vec' \
--rnn_type 'LSTM' \
--direction 'both' \
--wo_char \
--tok_emb 300 \
--cuda \
--save './models/random_500k_BiLSTM'

date

