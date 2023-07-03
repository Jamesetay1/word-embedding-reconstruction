#!/bin/bash
#SBATCH --job-name=ngram_dict
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc046
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ngram_dict_%j.log

pwd; hostname; date
module load pytorch/1.13.1-gpu
source /work/tc046/tc046/jamesetay1/subword-to-word/compact_reconstruction/venv/bin/activate

#!/usr/bin/bash
python src/preprocess/make_ngram_dic.py \
--ref_vec_path ../data/embeddings/crawl-300d-2M-subword.vec  \
--output tmp_out \
--n_max 3 \
--n_min 3

sort -k 2,2 -n -r tmp_out > resources/ngram_dics/ngram_dic.min3.max3

date
