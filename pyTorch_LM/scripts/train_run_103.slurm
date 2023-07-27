#!/usr/bin/bash
#SBATCH --job-name=RNNLM_Train_103
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=96:00:00
#SBATCH --account=tc046-jtaylor
#SBATCH --output=logs/training/%j.log

pwd; hostname; date
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python main.py \
--cuda \
--epochs 50 \
--pretrained_emb ../data/fastText_embeddings/fastTextS_r.vec \
--data ./data/wikitext-103 \
--freeze_emb

date