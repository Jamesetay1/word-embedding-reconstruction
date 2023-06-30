#!/usr/bin/bash

python src/inference.py \
--gpu 0 \
--model_path kvq_restricted_experiment/results/sep_kvq/1-1_20230614_12_27_09/model_epoch_300 \
--codecs_path kvq_restricted_experiment/resources/ngram_dict.min1.max1 \
--oov_word_path kvq_restricted_experiment/resources/to_reconstruct.txt

python src/inference.py \
--gpu 0 \
--model_path kvq_restricted_experiment/results/sep_kvq/1-3_20230614_12_27_09/model_epoch_300 \
--codecs_path kvq_restricted_experiment/resources/ngram_dict.min1.max3 \
--oov_word_path kvq_restricted_experiment/resources/to_reconstruct.txt

python src/inference.py \
--gpu 0 \
--model_path kvq_restricted_experiment/results/sep_kvq/1-5_20230614_12_27_09/model_epoch_300 \
--codecs_path kvq_restricted_experiment/resources/ngram_dict.min1.max5 \
--oov_word_path kvq_restricted_experiment/resources/to_reconstruct.txt

python src/inference.py \
--gpu 0 \
--model_path kvq_restricted_experiment/results/sep_kvq/1-10_20230614_12_27_27/model_epoch_300 \
--codecs_path kvq_restricted_experiment/resources/ngram_dict.min1.max10 \
--oov_word_path kvq_restricted_experiment/resources/to_reconstruct.txt

python src/inference.py \
--gpu 0 \
--model_path kvq_restricted_experiment/results/sep_kvq/1-20_20230614_12_27_27/model_epoch_300 \
--codecs_path kvq_restricted_experiment/resources/ngram_dict.min1.max20 \
--oov_word_path kvq_restricted_experiment/resources/to_reconstruct.txt

python src/inference.py \
--gpu 0 \
--model_path kvq_restricted_experiment/results/sep_kvq/1-30_20230614_12_27_27/model_epoch_300 \
--codecs_path kvq_restricted_experiment/resources/ngram_dict.min1.max30 \
--oov_word_path kvq_restricted_experiment/resources/to_reconstruct.txt

python src/inference.py \
--gpu 0 \
--model_path kvq_restricted_experiment/results/sep_kvq/3-30_20230614_12_27_27/model_epoch_300 \
--codecs_path kvq_restricted_experiment/resources/ngram_dict.min3.max30 \
--oov_word_path kvq_restricted_experiment/resources/to_reconstruct.txt