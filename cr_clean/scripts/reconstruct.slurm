#!/bin/bash
#SBATCH --job-name=james_reconstruction_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=1:00:00
#SBATCH --account=tc046-jtaylor
#SBATCH --output=logs/reconstruct/%j.log

pwd; hostname; date
export HOME=/work/tc046/tc046/jamesetay1
module load pytorch/1.13.1-gpu
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python src/inference.py \
--gpu 0 \
--model_path results/sep_kvq/20230724_18_09_33/model_epoch_300 \
--codecs_path resources/ngram_dic.max30.min3 \
--oov_word_path ../data/wordlist/500k.txt

date