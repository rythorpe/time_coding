#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --job-name=stp_rnn_sim
#SBATCH --partition=blanca-appm
#SBATCH --account=blanca-appm
#SBATCH --qos=blanca-appm
#SBATCH -o slurm.out
#SBATCH -e slurm.err

module purge
module load anaconda
conda activate rnn

python main_dev.py

conda deactivate
