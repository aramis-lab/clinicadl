#!/bin/bash
#SBATCH --partition=gpu_p1
#SBATCH --time=20:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1        # on réserve des coeurs physiques et non logiques
#SBATCH --ntasks=1
#SBATCH --workdir=/gpfswork/rech/zft/upd53tc/jobs/AD-DL/evaluation/patch
#SBATCH --output=./test10/pytorch_job_%j.out
#SBATCH --error=./test10/pytorch_job_%j.err
#SBATCH --job-name=test10_patch
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=mauricio.diaz@inria.fr

eval "$(conda shell.bash hook)"
conda activate clinicadl_env_py37

set -x
#export CUDA_VISIBLE_DEVICES=1
SCRIPT="evaluation_multiCNN.py"
SELECTION="best_acc"

# Data management
COHORT=$1
DIAGNOSES="sMCI pMCI"

BATCH=32
GPU=1
NUM_WORKERS=8
#WORKDIR="/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Zenodo_rerun/3_test"
OUTPUTDIR="patch3D_model-Conv4_FC3_preprocessing-linear_task-sMCI_pMCI_baseline_baseline-1_norm-1_t-1_multi-cnn_splits-5"

TSVPATH="$HOME/code/AD-DL/data/$COHORT/lists_by_diagnosis/test/"
RESULTSPATH="$SCRATCH/results/3_test/test10/$OUTPUTDIR/"
IMGPATH="$SCRATCH/../commun/datasets/${COHORT}"

if [ $COHORT = "ADNI" ]; then
IMGPATH="${IMGPATH}_rerun"
fi 

OPTIONS=""
if [ $GPU = 1 ]; then
OPTIONS="${OPTIONS} --gpu"
fi

python $HOME/code/AD-DL/clinicadl/clinicadl/patch_level/$SCRIPT \
  $IMGPATH \
  $TSVPATH \
  $RESULTSPATH \
  --selection $SELECTION \
  --dataset "test-$COHORT" \
  --diagnoses $DIAGNOSES \
  --num_workers $NUM_WORKERS \
  --selection_threshold 0.7 \
  --batch_size $BATCH \
  --prepare_dl \
  $OPTIONS
