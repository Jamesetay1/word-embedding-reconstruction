#!/bin/bash

#SBATCH --job-name=james_train_n1-1_6-12-2023
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/train%j.log

pwd; hostname; date
export HOME=/work/tc046/tc046/jamesetay1
module load pytorch/1.13.1-gpu
source /work/tc046/tc046/jamesetay1/subword-to-word/compact_reconstruction/venv/bin/activate

python src/train.py \
--gpu 0 \
--ref_vec_path resources/crawl-300d-2M-subword.vec \
--freq_path resources/freq_count.crawl-300d-2M-subword.txt \
--multi_hash two \
--maxlen 200 \
--n_max 30 \
--n_min 3 \
--codecs_path resources/ngram_dic.max30.min3 \
--network_type 3 \
--subword_type 4 \
--limit_size 1000000 \
--bucket_size 100000 \
--result_dir results/ \
--model_path models/ \
--hashed_idx \
--unique_false

date