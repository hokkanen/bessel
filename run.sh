#!/bin/bash -x
#SBATCH --account=Project_462000007
#SBATCH --partition=eap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1 
#SBATCH --time=00:15:00
#SBATCH --output=bessel.out
#SBATCH --error=bessel.err

# srun --account=Project_462000007 -N1 -n1 --partition=eap --cpus-per-task=1 --gpus-per-task=1 --time=00:15:00 ./bessel

srun bessel 1
#srun gdb -ex r -ex bt -ex quit --args bessel 2
