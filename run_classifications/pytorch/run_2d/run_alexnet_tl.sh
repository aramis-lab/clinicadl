#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=14-00:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch
#SBATCH --output=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/alexnet_tl_%j.out
#SBATCH --error=/network/lustre/iss01/home/junhao.wen/working_dir/pytorch/logs/alexnet_tl_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=junhao.wen@icm-institute.org
#SBATCH --job-name="alexnet tl"
#SBATCH --gres=gpu:1

## Load CUDA and python
module load python/2.7
module load CUDA
echo "Loading python module: $(which python)"



#export CUDA_VISIBLE_DEVICES=0
## Begin the training
echo "Begin the training:"
python /network/lustre/iss01/home/junhao.wen/Project/AD-DL/Code/pytorch/two_d_cnn/main_training.py --caps_directory /network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/train/AD_vs_CN.tsv --output_dir /network/lustre/iss01/home/junhao.wen/Project/AD-DL/Results/pytorch_alex_tl_fintune_alllayers_lr_10-7_dropout_0.8_bs32_epo50 --batch_size 32 --epochs 50 --learning_rate 1e-7 --network AlexNet --num_workers 8 --transfer_learning True --data_type from_slice --image_processing LinearReg --use_gpu True
echo "Finish!"

