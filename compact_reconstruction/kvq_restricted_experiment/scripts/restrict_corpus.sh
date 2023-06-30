#!/usr/bin/bash
python kvq_restricted_experiment/restrict_ngram_dicts.py \
--read_path=resources/freq_count.crawl-300d-2M-subword.txt \
--write_path=kvq_restricted_experiment/resources/restricted_freq_counts.txt
