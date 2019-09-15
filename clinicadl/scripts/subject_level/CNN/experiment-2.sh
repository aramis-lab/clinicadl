#!/bin/bash
#SBATCH --partition=gpu_gct3
#SBATCH --time=20:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --chdir=.
#SBATCH --output=pytorch_job_%j.out
#SBATCH --error=pytorch_job_%j.err
#SBATCH --job-name=3DAE
#SBATCH --gres=gpu:1

#export http_proxy=http://10.10.2.1:8123
#export https_proxy=http://10.10.2.1:8123

# Experiment taining CNN
eval "$(conda shell.bash hook)"
conda activate clinicadl_env_py37

# Network structure
NETWORK="Conv5_FC3"
COHORT="ADNI"
DATE="reproducibility_results"

# Input arguments to clinicadl
CAPS_DIR="$SCRATCH/../commun/datasets/ADNI_rerun"
TSV_PATH="$HOME/code/AD-DL/data/$COHORT/lists_by_diagnosis/train"
OUTPUT_DIR="$SCRATCH/results_subject/$DATE"

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
NORMALIZATION=1
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
  --minmaxnormalization \
  --split $SPLIT \
  --accumulation_steps $ACCUMULATION \
  --epochs $EPOCHS \
  --sampler $SAMPLER \
  --learning_rate $LR \
  --patience $PATIENCE
