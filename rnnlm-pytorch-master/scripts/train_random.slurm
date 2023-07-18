#!/usr/bin/bash
#SBATCH --job-name=RNNLM_random
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --account=tc046-jtaylor
#SBATCH --output=logs/training/%j.log

pwd; hostname; date
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python train.py \
--rnn_type 'LSTM' \
--direction 'left2right' \
--wo_char \
--glove '../data/random_embeddings/random500k.vec' \
--tok_emb 300 \
--cuda \
--save './models/random_500k_f' \
--dict './models/random_500k_f_dict.pkl'

date

