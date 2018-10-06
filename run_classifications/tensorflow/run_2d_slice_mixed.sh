#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=14-00:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=/network/lustre/iss01/home/junhao.wen/working_dir
#SBATCH --output=/network/lustre/iss01/home/junhao.wen/working_dir/logs/run_2dcnn_%j.out
#SBATCH --error=/network/lustre/iss01/home/junhao.wen/working_dir/logs/run_2dcnn_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=junhao.wen@icm-institute.org
#SBATCH --job-name=training 2dcnn
#SBATCH --comment="train 2dcnn with ADNI"
#SBATCH --gres=gpu:2

## Load CUDA and python with tensorflow. Ensure Maximum install cuDNN
module load python/2.7

module load CUDA/9.0

## Begin the training
echo "Begin the training:"
python /network/lustre/iss01/home/junhao.wen/Project/AD-DL/run_2DLenet_slices_mixed.py
echo "Finish!"


