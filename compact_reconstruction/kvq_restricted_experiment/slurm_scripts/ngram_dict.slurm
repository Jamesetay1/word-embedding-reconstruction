#!/bin/bash
#SBATCH --job-name=ngram_dict_restricted
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=tc046
#SBATCH --gres=gpu:1
#SBATCH --output=kvq_restricted_experiment/results/logs/ngram_dict_restricted%j.log


pwd; hostname; date
module load pytorch/1.13.1-gpu
source /work/tc046/tc046/jamesetay1/subword-to-word/compact_reconstruction/venv/bin/activate

srun kvq_restricted_experiment/scripts/ngram_dict.sh
date
