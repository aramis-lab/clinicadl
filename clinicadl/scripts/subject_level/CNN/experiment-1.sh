#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --mem=17G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --chdir=.
#SBATCH --output=outputs/pytorch_job_%j.out
#SBATCH --error=outputs/pytorch_job_%j.err
#SBATCH --job-name=3DAE
#SBATCH --gres=gpu:1

#export http_proxy=http://10.10.2.1:8123
#export https_proxy=http://10.10.2.1:8123

# Experiment taining CNN
module load clinica.all
eval "$(conda shell.bash hook)"
conda activate clinica_dl_pre_py36

# Network structure
NETWORK="Conv5_FC3"
COHORT="ADNI"
DATE="reproducibility_results"

# Input arguments to clinicadl
CAPS_DIR="/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_rerun"
TSV_PATH="/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/code/AD-DL/data/$COHORT/lists_by_diagnosis/train"
OUTPUT_DIR="/network/lustre/iss01/home/mauricio.diazmelo/ADNI_rerun/results/$DATE/"

# Computation ressources
NUM_PROCESSORS=8

# Dataset Management
PREPROCESSING='linear'
BASELINE=1
TASK='AD CN'
SPLITS=5
SPLIT=$1

# Training arguments
EPOCHS=50
BATCH=6
ACCUMULATION=2
EVALUATION=20
SAMPLER="random"
LR=1e-4
NORMALIZATION=0
PATIENCE=5
TOLERANCE=0

OPTIONS=""

TASK_NAME="${TASK// /_}"

if [ $BASELINE = 1 ]; then
  echo "using only baseline data"
  TASK_NAME="${TASK_NAME}_baseline"
  OPTIONS="$OPTIONS --baseline"
  PATIENCE=10
fi
echo $TASK_NAME

NAME="model-${NETWORK}_preprocessing-${PREPROCESSING}_task-${TASK_NAME}_norm-${NORMALIZATION}"

if [ $SPLITS > 0 ]; then
echo "Use of $SPLITS-fold cross validation, split $SPLIT"
NAME="${NAME}_splits-${SPLITS}"
fi

echo $NAME

# Run clinicadl
clinicadl train \
  subject \
  $CAPS_DIR \
  $TSV_PATH \
  $OUTPUT_DIR$NAME \
  $NETWORK \
  --diagnoses $TASK \
  --use_gpu \
  --nproc $NUM_PROCESSORS \
  --batch_size $BATCH \
  --evaluation_steps $EVALUATION \
  --preprocessing $PREPROCESSING \
  --baseline \
  --n_splits $SPLITS \
  --split $SPLIT \
  --accumulation_steps $ACCUMULATION \
  --epochs $EPOCHS \
  --sampler $SAMPLER \
  --learning_rate $LR \
  --patience $PATIENCE
