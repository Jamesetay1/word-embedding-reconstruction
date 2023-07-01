#!/usr/bin/bash
#SBATCH --job-name=CNN_reconstruct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --output=logs/reconstruction/%j.log

pwd; hostname; date
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python src/reconstruct.py \
--model_path checkpoints/500k_50c_best.ckpt \
--testset_path preprocessed/500k.npy \
--reconstructed_path results/model500k_50c_reconstructed500k.vec \
--charset_path preprocessed/restricted_charset \
--batch_size 50 \
--filter_interval 50

date