#!/usr/bin/bash
python kvq_restricted_experiment/restrict_ngram_dicts.py \
--read_path=resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max1 \
--write_path=kvq_restricted_experiment/resources/ngram_dict.min1.max1

python kvq_restricted_experiment/restrict_ngram_dicts.py \
--read_path=resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max3 \
--write_path=kvq_restricted_experiment/resources/ngram_dict.min1.max3

python kvq_restricted_experiment/restrict_ngram_dicts.py \
--read_path=resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max5 \
--write_path=kvq_restricted_experiment/resources/ngram_dict.min1.max5

python kvq_restricted_experiment/restrict_ngram_dicts.py \
--read_path=resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max10 \
--write_path=kvq_restricted_experiment/resources/ngram_dict.min1.max10

python kvq_restricted_experiment/restrict_ngram_dicts.py \
--read_path=resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max20 \
--write_path=kvq_restricted_experiment/resources/ngram_dict.min1.max20

python kvq_restricted_experiment/restrict_ngram_dicts.py \
--read_path=resources/kvq_restricted_ngram_dicts/ngram_dict.min1.max30 \
--write_path=kvq_restricted_experiment/resources/ngram_dict.min1.max30

python kvq_restricted_experiment/restrict_ngram_dicts.py \
--read_path=resources/kvq_restricted_ngram_dicts/ngram_dict.min3.max30 \
--write_path=kvq_restricted_experiment/resources/ngram_dict.min3.max30

