#!/bin/bash

#SBATCH --job-name=SAM3-6_Training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --account=tc046-jtaylor
#SBATCH --output=logs/training/%j.log

pwd; hostname; date
export HOME=/work/tc046/tc046/jamesetay1
module load pytorch/1.13.1-gpu
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate


date
python src/train.py \
--gpu 0 \
--ref_vec_path ../data/fastText_embeddings/fastText.vec \
--freq_path ../data/frequency_counts_F.txt \
--multi_hash two \
--maxlen 200 \
--n_max 6 \
--n_min 3 \
--codecs_path ../data/ngram_dicts/ngram_dict.min3.max6_F \
--network_type 3 \
--subword_type 4 \
--limit_size 1000000 \
--bucket_size 100000 \
--result_dir results/ \
--model_path models/ \
--hashed_idx \
--unique_false

date