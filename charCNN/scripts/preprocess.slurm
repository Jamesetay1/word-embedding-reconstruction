#!/usr/bin/bash
#SBATCH --job-name=james_preprocess
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --time=5:00:00
#SBATCH --output=logs/%j.log

pwd; hostname; date
export HOME=/work/tc046/tc046/jamesetay1
source /work/tc046/tc046/jamesetay1/subword-to-word/venv/bin/activate

date
srun scripts/preprocess.sh
date