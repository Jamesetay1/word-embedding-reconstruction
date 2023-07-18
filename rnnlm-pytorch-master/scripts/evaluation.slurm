#!/usr/bin/bash
#SBATCH --job-name=RRNLM_evaluation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=05:00:00
#SBATCH --account=tc046-jtaylor
#SBATCH --output=logs/evaluation/%j.log

pwd; hostname; date
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

#python evaluation.py \
#--load './models/charCNN_500k_f' \
#--outf './output/charCNN_temp.txt' \
#--dict './models/charCNN_500k_f_dict.pkl' \
#--cuda

#python evaluation.py \
#--load './models/fastText_500k_f' \
#--outf './output/fastText_temp.txt' \
#--dict './models/fastText_500k_f_dict.pkl' \
#--cuda

#python evaluation.py \
#--load './models/OH' \
#--outf './output/OH_temp.txt' \
#--dict './models/OH_dict.pkl' \
#--cuda

python evaluation.py \
--load './models/zero_500k_f' \
--outf './output/zero_temp.txt' \
--dict './models/zero_500k_f_dict.pkl' \
--cuda

date