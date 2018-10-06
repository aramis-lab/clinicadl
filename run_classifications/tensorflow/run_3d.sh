#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=02-00:00:00
#SBATCH --mem=36G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=/network/lustre/iss01/home/junhao.wen/working_dir
#SBATCH --output=/network/lustre/iss01/home/junhao.wen/working_dir/logs/run_3dcnn_%j.out
#SBATCH --error=/network/lustre/iss01/home/junhao.wen/working_dir/logs/run_3dcnn_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=junhao.wen@icm-institute.org
#SBATCH --job-name=training 3dcnn
#SBATCH --comment="train 3dcnn with ADNI"
#SBATCH --exclusive
#SBATCH --gres=gpu:1

## Load CUDA and python with tensorflow. Ensure Maximum install cuDNN
module load python/2.7
module load CUDA/9.0
module load clinica/05_04_2018

## Begin the training
echo "Begin the training:"
python /network/lustre/iss01/home/junhao.wen/Project/AD-DL/run_3DCNN.py
echo "Finish!"


