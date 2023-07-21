#!/usr/bin/bash
#SBATCH --job-name=RNNLM_Train_103
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
--pretrained_emb ../data/random_embeddings/zero500k.vec \
--data ./data/wikitext-103 \
--freeze_emb

python main.py \
--cuda \
--epochs 50 \
--pretrained_emb ../data/random_embeddings/random500k.vec \
--data ./data/wikitext-103 \
--freeze_emb

python main.py \
--cuda \
--epochs 50 \
--pretrained_emb ../charCNN/results/model500k_50c_reconstructed500k.vec \
--data ./data/wikitext-103 \
--freeze_emb

python main.py \
--cuda \
--epochs 50 \
--pretrained_emb ../data/fastText_embeddings/500k.vec \
--data ./data/wikitext-103 \
--freeze_emb

date