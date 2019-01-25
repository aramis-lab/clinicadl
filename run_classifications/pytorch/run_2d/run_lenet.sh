#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=14-00:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch
#SBATCH --output=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/run_2dcnn_scratch_%j.out
#SBATCH --error=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/run_2dcnn_scratch_%j.err
#SBATCH --job-name="training 2dcnn from scratch"
#SBATCH --gres=gpu:1

## Load CUDA and python
module load python/2.7
module load CUDA
echo "Loading python module: $(which python)"



#export CUDA_VISIBLE_DEVICES=0
## Begin the training
echo "Begin the training:"
python /network/lustre/iss01/home/junhao.wen/Project/AD-DL/Code/pytorch/two_d_cnn/main_training.py --caps_directory /network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/train/AD_vs_CN_baseline.tsv --output_dir /network/lustre/iss01/home/junhao.wen/Project/AD-DL/Results/pytorch_from_scract_lenet_BS64_Epoch20_LR_10-4_ADAM_dropout0.8_wd_0.01  --batch_size 64 --epochs 20 --learning_rate 1e-4 --num_workers 8 --transfer_learning False 
echo "Finish!"


