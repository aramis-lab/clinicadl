#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=1:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=.
#SBATCH --output=outputs/scratch_baseline_job_%j.out
#SBATCH --error=outputs/scratch_baseline_job_%j.err
#SBATCH --job-name=SVM

export http_proxy=http://10.10.2.1:8123
export https_proxy=http://10.10.2.1:8123

module load pytorch/1.0.0
# Network structure
SCRIPT="evaluation"

# Dataset Management
COHORT='ADNI'
CAPS_EXT="_clinica_spm"
BASELINE=0
TASK='sMCI pMCI'
SPLITS=5
SET="validation"

NUM_WORKERS=16
DATE="refactoring_results/svm4"

TSVPATH="/network/lustre/iss01/home/elina.thibeausutre/data/Frontiers/$COHORT/lists_by_diagnosis/train/"
RESULTSPATH="/network/lustre/iss01/home/elina.thibeausutre/results/$DATE/"
IMGPATH="/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/$COHORT$CAPS_EXT"

TASK_NAME="train_AD_CN"
TASK_NAME="${TASK_NAME}_baseline-${BASELINE}_final"
echo $TASK_NAME

NAME="model-svm_task-${TASK_NAME}"

cd /network/lustre/iss01/home/elina.thibeausutre/Code/AD-DL/clinicadl
pwd
python -m clinicadl.svm.$SCRIPT $TSVPATH $IMGPATH $IMGPATH $RESULTSPATH$NAME -w $NUM_WORKERS -d $TASK $OPTIONS --set $SET --train_mode
