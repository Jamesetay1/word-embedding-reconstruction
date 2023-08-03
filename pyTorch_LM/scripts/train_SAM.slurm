#!/usr/bin/bash
#SBATCH --job-name=RNNLM_SAM
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=5:00:00
#SBATCH --account=tc046-jtaylor
#SBATCH --output=logs/training/%j.log

pwd; hostname; date
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python main.py \
--cuda \
--epochs 50 \
--pretrained_emb ../data/reconstructed_embeddings/SAM-100.vec \
--freeze_emb

python main.py \
--cuda \
--epochs 50 \
--pretrained_emb ../data/reconstructed_embeddings/SAM-100+3.vec \
--freeze_emb

python main.py \
--cuda \
--epochs 50 \
--pretrained_emb ../data/reconstructed_embeddings/SAM-100+4.vec \
--freeze_emb

date