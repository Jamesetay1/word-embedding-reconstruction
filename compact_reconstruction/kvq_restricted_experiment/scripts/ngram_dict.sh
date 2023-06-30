#!/usr/bin/bash
python src/preprocess/make_ngram_dic.py \
--ref_vec_path ../data/embeddings/crawl-300d-2M-subword.vec  \
--output kvq_restricted_experiment/tmp.out \
--n_max 1 \
--n_min 1

sort -k 2,2 -n -r kvq_restricted_experiment/tmp.out  > resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max1


python src/preprocess/make_ngram_dic.py \
--ref_vec_path ../data/embeddings/crawl-300d-2M-subword.vec  \
--output kvq_restricted_experiment/tmp.out \
--n_max 3 \
--n_min 1

sort -k 2,2 -n -r kvq_restricted_experiment/tmp.out  > resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max3


python src/preprocess/make_ngram_dic.py \
--ref_vec_path ../data/embeddings/crawl-300d-2M-subword.vec  \
--output kvq_restricted_experiment/tmp.out \
--n_max 5 \
--n_min 1

sort -k 2,2 -n -r kvq_restricted_experiment/tmp.out  > resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max5


python src/preprocess/make_ngram_dic.py \
--ref_vec_path ../data/embeddings/crawl-300d-2M-subword.vec  \
--output kvq_restricted_experiment/tmp.out  \
--n_max 10 \
--n_min 1

sort -k 2,2 -n -r kvq_restricted_experiment/tmp.out  > resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max10


python src/preprocess/make_ngram_dic.py \
--ref_vec_path ../data/embeddings/crawl-300d-2M-subword.vec  \
--output kvq_restricted_experiment/tmp.out  \
--n_max 20 \
--n_min 1

sort -k 2,2 -n -r kvq_restricted_experiment/tmp.out  > resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max20

python src/preprocess/make_ngram_dic.py \
--ref_vec_path ../data/embeddings/crawl-300d-2M-subword.vec  \
--output kvq_restricted_experiment/tmp.out  \
--n_max 30 \
--n_min 1

sort -k 2,2 -n -r kvq_restricted_experiment/tmp.out  > resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max30

python src/preprocess/make_ngram_dic.py \
--ref_vec_path ../data/embeddings/crawl-300d-2M-subword.vec  \
--output kvq_restricted_experiment/tmp.out  \
--n_max 30 \
--n_min 3

sort -k 2,2 -n -r kvq_restricted_experiment/tmp.out  > resources/kvq_restricted_ngram_dicts/ngram_dict.min3.max30