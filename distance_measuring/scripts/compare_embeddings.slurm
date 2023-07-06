#!/usr/bin/bash
#SBATCH --job-name=compare_embeddings
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --account=tc046-jtaylor
#SBATCH --output=distance_measuring/logs/%j.log

pwd; hostname; date
#module load python/3.10.8-gpu
#module load nvidia/cudnn/8.6.0-cuda-11.6

source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python distance_measuring/compare_embeddings.py \
--reconstructed_path compact_reconstruction/results/sep_kvq/n3-6_mhtwo/embedding_epoch300/embedding_reconstruct_original.txt \
--reference_path data/fastText_embeddings/10k.vec \
--top_n 10000 \
--cossim \
--nearest 10 \
--wordsim

date