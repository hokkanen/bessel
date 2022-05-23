#!/bin/bash -x
#SBATCH --account=Project_2002078
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100:4
#SBATCH --time=00:15:00
#SBATCH --output=bessel.out
#SBATCH --error=bessel.err

# salloc --account=Project_2002078 --nodes=1 --partition=gputest --gres=gpu:v100:4 --mem-per-cpu=16G --time=00:15:00

srun bessel 4
#srun gdb -ex r -ex bt -ex quit --args bessel 2
