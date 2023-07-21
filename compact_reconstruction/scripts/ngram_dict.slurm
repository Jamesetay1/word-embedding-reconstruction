#!/bin/bash
#SBATCH --job-name=ngram_dict
#SBATCH --time=05:00:00
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=tc046-jtaylor
#SBATCH --output=logs/ngrams/%j.log

pwd; hostname; date
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

#!/usr/bin/bash
python src/preprocess/make_ngram_dic.py \
--ref_vec_path ../data/fastText_embeddings/fastText.vec  \
--output tmp_out \
--n_max 6 \
--n_min 3

sort -k 2,2 -n -r tmp_out > ../data/ngram_dicts/ngram_dict.min3.max6_F
date
