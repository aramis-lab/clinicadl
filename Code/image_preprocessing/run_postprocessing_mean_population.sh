#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=4-00:00:00
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --nodes=1
#SBATCH --workdir=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch
#SBATCH --output=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/image_postprocessing_mean_img_%j.out
#SBATCH --error=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/image_postprocessing_mean_img_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=junhao.wen@icm-institute.org
#SBATCH --job-name="image postprocessing"

## Load CUDA and python
module load python/2.7
module load CUDA/9.0
#export CUDA_VISIBLE_DEVICES=0,1
## Begin the training
echo "Begin the image postprocessing:"
python /network/lustre/iss01/home/junhao.wen/Project/AD-DL/Code/image_preprocessing/run_postprocessing_mean_image_population.py
echo "Finish!"


