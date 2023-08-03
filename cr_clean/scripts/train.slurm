#!/bin/bash

#SBATCH --job-name=cr_clean_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=12:00:00
#SBATCH --account=tc046-jtaylor
#SBATCH --output=logs/training/%j.log

pwd; hostname; date
export HOME=/work/tc046/tc046/jamesetay1
module load pytorch/1.13.1-gpu
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

date
python src/train.py \
--gpu 0 \
--ref_vec_path ../data/fastText_embeddings/fastTextS.vec \
--freq_path resources/frequency.txt \
--multi_hash two \
--maxlen 200 \
--codecs_path resources/ngram_dic.max30.min3 \
--network_type 3 \
--subword_type 4 \
--limit_size 500 \
--bucket_size 100 \
--result_dir results/ \
--hashed_idx \
--unique_false

date