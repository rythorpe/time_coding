#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem=250G
#SBATCH --job-name=stp_rnn_sim
#SBATCH --partition=blanca-appm
#SBATCH --account=blanca-appm
#SBATCH --qos=blanca-appm
#SBATCH -o slurm_size_control.out
#SBATCH -e slurm_size_control.err

module purge
module load anaconda
conda activate rnn

echo "== Start of Job =="
python param_sweep_size_control.py
echo "== End of Job =="

conda deactivate