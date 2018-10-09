#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=14-00:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch
#SBATCH --output=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/run_2dcnn_%j.out
#SBATCH --error=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/run_2dcnn_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=junhao.wen@icm-institute.org
#SBATCH --job-name="training 2dcnn"
#SBATCH --gres=gpu:2

## Load CUDA and python
module load python/2.7
module load CUDA/9.0
#export CUDA_VISIBLE_DEVICES=0,1
## Begin the training
echo "Begin the training:"
python /network/lustre/iss01/home/junhao.wen/Project/AD-DL/Code/pytorch/two_d_cnn/main.py
echo "Finish!"


