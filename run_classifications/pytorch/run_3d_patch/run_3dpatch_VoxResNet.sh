#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=14-00:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch
#SBATCH --output=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/run_3dpatch_%j.out
#SBATCH --error=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/run_3dpatch_%j.err
#SBATCH --job-name="training 3d patch"
#SBATCH --gres=gpu:1

## Load CUDA and python
module load python/2.7
module load CUDA
echo "Loading python module: $(which python)"



#export CUDA_VISIBLE_DEVICES=0
## Begin the training
echo "Begin the training:"
python /network/lustre/iss01/home/junhao.wen/Project/AD-DL/Code/pytorch/three_d_cnn/patch_level/main_training.py --caps_directory /network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/train/AD_vs_CN.tsv --output_dir /network/lustre/iss01/home/junhao.wen/Project/AD-DL/Results/pytorch_3dpatch_VoxResNet_bs16_lr0.001_epoch100 --batch_size 16 --use_gpu True --epochs 100 --network VoxResNet --patch_size 51 --patch_stride 51 --num_workers 4
echo "Finish!"


