#!/usr/bin/bash
#SBATCH --job-name=RNNLM_charCNN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --account=tc046-jtaylor
#SBATCH --output=logs/training/%j.log

pwd; hostname; date
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python train.py \
--glove '../data/fastText_embeddings/500k.vec' \
--rnn_type 'LSTM' \
--direction 'left2right' \
--wo_char \
--tok_emb 300 \
--cuda \
--save './models/fastText_500k_LSTM' \
--dict './models/fastText_500k_LSTM_dict.pkl'

date