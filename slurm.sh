#!/bin/bash
#SBATCH --job-name=gpu-pagerank
#SBATCH --output=output_pagerank.txt
#SBATCH --error=error_pagerank.txt
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100

# Load CUDA module
# module purge
# module load gcc/11.5
# module load StdEnv
# module load cuda/12.2.0

# Compile
# nvcc *.cu -o out

srun "$@"