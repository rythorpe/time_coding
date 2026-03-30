#!/bin/bash

#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem=200G
#SBATCH --job-name=stp_rnn_sim
#SBATCH --partition=blanca-appm
#SBATCH --account=blanca-appm
#SBATCH --qos=blanca-appm
#SBATCH -o slurm.out
#SBATCH -e slurm.err

module purge
module load anaconda
conda activate rnn

echo "== Start of Job =="
python param_sweep_size_control.py
echo "== End of Job =="

conda deactivate