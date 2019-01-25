#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=14-00:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch
#SBATCH --output=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/run_2dresnet_%j.out
#SBATCH --error=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/run_2dresnet_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=junhao.wen@icm-institute.org
#SBATCH --job-name="training 2dresnet transfer learning"
#SBATCH --gres=gpu:1

## Load CUDA and python
module load python/2.7
module load CUDA
echo "Loading python module: $(which python)"



#export CUDA_VISIBLE_DEVICES=0
## Begin the training
echo "Begin the training:"
python /network/lustre/iss01/home/junhao.wen/Project/AD-DL/Code/pytorch/two_d_cnn/main_training.py --caps_directory /network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/train/AD_vs_CN_baseline.tsv --output_dir /network/lustre/iss01/home/junhao.wen/Project/AD-DL/Results/pytorch_resnet18_tl_fintune_lastResBlock_1fc_dropout0.8_lr10-7_bs32_ep80_normalizedmeanstd --batch_size 32 --epochs 80 --learning_rate 1e-7 --num_workers 8 --network ResNet2D --transfer_learning True --image_processing LinearReg
echo "Finish!"

