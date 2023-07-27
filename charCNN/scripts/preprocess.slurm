#!/usr/bin/bash
#SBATCH --job-name=james_preprocess
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --output=logs/%j.log
#SBATCH --account=tc046-jtaylor

pwd; hostname; date
export HOME=/work/tc046/tc046/jamesetay1
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

date
python src/preprocess.py \
--vec_path ../data/fastText_embeddings/fastTextS_r.vec \
--charset_path preprocessed/restricted_charset \
--npy_path preprocessed/fastTextS_r.npy

date