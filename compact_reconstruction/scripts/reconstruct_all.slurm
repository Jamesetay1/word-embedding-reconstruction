#!/bin/bash
#SBATCH --job-name=james_reconstruction_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --output=results/logs/reconstruct_all_%j.log

pwd; hostname; date
export HOME=/work/tc046/tc046/jamesetay1
module load pytorch/1.13.1-gpu
source /work/tc046/tc046/jamesetay1/subword-to-word/compact_reconstruction/venv/bin/activate

python src/save_embedding.py \
--gpu 0 \
--inference \
--model_path results/sep_kvq/KVQ-FH_1-1/model_epoch_300

date