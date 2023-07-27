#!/usr/bin/bash
#SBATCH --job-name=compare_embeddings
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=1:00:00
#SBATCH --account=tc046-jtaylor
#SBATCH --output=distance_measuring/logs/%j.log

pwd; hostname; date
#module load python/3.10.8-gpu
#module load nvidia/cudnn/8.6.0-cuda-11.6

source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python distance_measuring/compare_embeddings.py \
--reconstructed_path data/reconstructed_embeddings/charCNN.vec \
--reference_path data/fastText_embeddings/fastTextS_r.vec \
--top_n 10000 \
--cossim \
--nearest 10 \
--wordsim

python distance_measuring/compare_embeddings.py \
--reconstructed_path data/reconstructed_embeddings/charCNN.vec \
--reference_path data/fastText_embeddings/fastTextS_r.vec \
--top_n 10000 \
--cossim \
--nearest 100 \
--wordsim

python distance_measuring/compare_embeddings.py \
--reconstructed_path data/reconstructed_embeddings/SAM3-30+1.vec \
--reference_path data/fastText_embeddings/fastTextS_r.vec \
--top_n 10000 \
--cossim \
--nearest 10 \
--wordsim

python distance_measuring/compare_embeddings.py \
--reconstructed_path data/reconstructed_embeddings/SAM3-30+1.vec \
--reference_path data/fastText_embeddings/fastTextS_r.vec \
--top_n 10000 \
--cossim \
--nearest 100 \
--wordsim

python distance_measuring/compare_embeddings.py \
--reconstructed_path data/reconstructed_embeddings/SAM3-30+2.vec \
--reference_path data/fastText_embeddings/fastTextS_r.vec \
--top_n 10000 \
--cossim \
--nearest 10 \
--wordsim

python distance_measuring/compare_embeddings.py \
--reconstructed_path data/reconstructed_embeddings/SAM3-30+2.vec \
--reference_path data/fastText_embeddings/fastTextS_r.vec \
--top_n 10000 \
--cossim \
--nearest 100 \
--wordsim

python distance_measuring/compare_embeddings.py \
--reconstructed_path data/reconstructed_embeddings/SAM3-30+3.vec \
--reference_path data/fastText_embeddings/fastTextS_r.vec \
--top_n 10000 \
--cossim \
--nearest 10 \
--wordsim

python distance_measuring/compare_embeddings.py \
--reconstructed_path data/reconstructed_embeddings/SAM3-30+3.vec \
--reference_path data/fastText_embeddings/fastTextS_r.vec \
--top_n 10000 \
--cossim \
--nearest 100 \
--wordsim

python distance_measuring/compare_embeddings.py \
--reconstructed_path data/reconstructed_embeddings/SAM3-30+4.vec \
--reference_path data/fastText_embeddings/fastTextS_r.vec \
--top_n 10000 \
--cossim \
--nearest 10 \
--wordsim

python distance_measuring/compare_embeddings.py \
--reconstructed_path data/reconstructed_embeddings/SAM3-30+4.vec \
--reference_path data/fastText_embeddings/fastTextS_r.vec \
--top_n 10000 \
--cossim \
--nearest 100 \
--wordsim

date