#!/bin/bash
#SBATCH --partition=gpu_p1
#SBATCH --time=5:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1        # on réserve des coeurs physiques et non logiques
#SBATCH --ntasks=1
#SBATCH --workdir=/gpfswork/rech/zft/upd53tc/jobs2/AD-DL/tests/evaluation/slice_level
#SBATCH --output=./pytorch_job_%j.out
#SBATCH --error=./pytorch_job_%j.err
#SBATCH --job-name=test13_slice
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=mauricio.diaz@inria.fr

eval "$(conda shell.bash hook)"
conda activate clinicadl_env_py37

set -x
#export CUDA_VISIBLE_DEVICES=1
SCRIPT="evaluation_test.py"
SELECTION="best_acc"

# Data management
COHORT="OASIS_atrophy"
DIAGNOSES="AD CN"

BATCH=32
GPU=1
NUM_WORKERS=8
OUTPUTDIR="slice2D_model-resnet18_preprocessing-linear_task-AD-CN_baseline-0_preparedl-1_splits-5"

TSVPATH="$HOME/code/AD-DL/data/$COHORT/lists_by_diagnosis/test/"
RESULTSPATH="$SCRATCH/results/trivial_tests_2/3_test/$OUTPUTDIR/"
IMGPATH="$SCRATCH/../commun/datasets/${COHORT}"

if [ $COHORT = "ADNI" ]; then
IMGPATH="${IMGPATH}_rerun"
fi 

OPTIONS=""
if [ $GPU = 1 ]; then
OPTIONS="${OPTIONS} --gpu"
fi

python $HOME/code/AD-DL/clinicadl/clinicadl/slice_level/$SCRIPT \
  $IMGPATH \
  $TSVPATH \
  $RESULTSPATH \
  --dataset "test-$COHORT" \
  --diagnoses $DIAGNOSES \
  --selection $SELECTION \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH \
  $OPTIONS
