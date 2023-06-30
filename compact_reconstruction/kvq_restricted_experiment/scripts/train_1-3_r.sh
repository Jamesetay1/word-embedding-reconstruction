#!/usr/bin/bash

python src/train.py \
--gpu 0 \
--ref_vec_path kvq_restricted_experiment/resources/restricted_vectors.vec \
--freq_path kvq_restricted_experiment/resources/restricted_freq_counts.txt \
--multi_hash two \
--maxlen 200 \
--n_max 3 \
--n_min 1 \
--codecs_path kvq_restricted_experiment/resources/ngram_dict.min1.max3 \
--network_type 3 \
--subword_type 4 \
--limit_size 1000000 \
--bucket_size 100000 \
--result_dir kvq_restricted_experiment/results \
--hashed_idx \
--unique_false