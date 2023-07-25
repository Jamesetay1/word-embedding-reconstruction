#!/usr/bin/bash
#SBATCH --job-name=RNNLM_OH
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=1:00:00
#SBATCH --account=tc046-jtaylor
#SBATCH --output=logs/training/%j.log

pwd; hostname; date
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

python main.py \
--cuda \
--epochs 50 \
--pretrained_emb ../cr_clean/results/sep_kvq/20230724_18_09_33/inference_embedding_epoch300/embedding.txt \
--freeze_emb

date