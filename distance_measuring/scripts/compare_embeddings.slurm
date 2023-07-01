#!/usr/bin/bash
#SBATCH --job-name=CNN_500K
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --output=logs/training/%j.log

pwd; hostname; date
#module load python/3.10.8-gpu
#module load nvidia/cudnn/8.6.0-cuda-11.6

source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python src/compare_embeddings.py \
--reconstructed_path data/1k.vec \
--reference_path data/1k.vec \
--top_n 500 \
--cossim \
--nearest 10 \
--wordsim

date