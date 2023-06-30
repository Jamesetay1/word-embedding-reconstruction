#!/bin/bash
#SBATCH --job-name=james_reconstruction_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --output=results/logs/reconstruct%j.log

pwd; hostname; date
export HOME=/work/tc046/tc046/jamesetay1
module load pytorch/1.13.1-gpu
source /work/tc046/tc046/jamesetay1/subword-to-word/compact_reconstruction/venv/bin/activate

python src/inference.py \
--gpu 0 \
--model_path results/sep_kvq/3-3_20230615_11_22_18/model_epoch_300 \
--codecs_path resources/ngram_dics/ngram_dic.min3.max3 \
--oov_word_path ../data/google-10000-english.txt

date