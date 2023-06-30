#!/usr/bin/bash
python kvq_restricted_experiment/restrict_ngram_dicts.py \
--read_path=kvq_restricted_experiment/resources/restricted_freq_counts.txt \
--write_path=kvq_restricted_experiment/resources/to_reconstruct.txt \
--drop_numbers