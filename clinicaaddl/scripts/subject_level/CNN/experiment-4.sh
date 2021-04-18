#!/bin/bash
#SBATCH --partition=gpu_p1
#SBATCH --time=20:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=/gpfswork/rech/zft/upd53tc/jobs2/AD-DL/train/subject_level/cnn
#SBATCH --output=./exp4/pytorch_job_%j.out
#SBATCH --error=./exp4/pytorch_job_%j.err
#SBATCH --job-name=3CNN_subject
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --mail-type=END
#SBATCH --mail-user=mauricio.diaz-melo@inria.fr

#export http_proxy=http://10.10.2.1:8123
#export https_proxy=http://10.10.2.1:8123

# Experiment taining CNN
eval "$(conda shell.bash hook)"
conda activate clinicadl_env_py37

# Network structure
NETWORK="Conv5_FC3"
COHORT="ADNI"
DATE="reproducibility_results_2"

# Input arguments to clinicaaddl
CAPS_DIR="$SCRATCH/../commun/datasets/${COHORT}_rerun"
TSV_PATH="$HOME/code/AD-DL/data/$COHORT/lists_by_diagnosis/train"
OUTPUT_DIR="$SCRATCH/results/$DATE/"

# Computation ressources
NUM_PROCESSORS=8
GPU=1

# Dataset Management
PREPROCESSING='linear'
TASK='AD CN'
BASELINE=0
SPLITS=5
SPLIT=$SLURM_ARRAY_TASK_ID

# Training arguments
EPOCHS=50
BATCH=6
ACCUMULATION=2
EVALUATION=20
LR=1e-4
WEIGHT_DECAY=0
NORMALIZATION=1
PATIENCE=5
TOLERANCE=0

# Pretraining
T_BOOL=1
T_PATH="subject_model-Conv5_FC3_preprocessing-linear_task-autoencoder_baseline-1_norm-1_splits-5"
T_PATH="$SCRATCH/results/$DATE/$T_PATH"
T_DIFF=0

# Other options
OPTIONS=""

if [ $GPU = 1 ]; then
OPTIONS="${OPTIONS} --use_gpu"
fi

if [ $NORMALIZATION = 1 ]; then
OPTIONS="${OPTIONS} --minmaxnormalization"
fi

if [ $T_BOOL = 1 ]; then
OPTIONS="$OPTIONS --transfer_learning_autoencoder --transfer_learning_path $T_PATH"
fi
TASK_NAME="${TASK// /_}"

if [ $BASELINE = 1 ]; then
  echo "using only baseline data"
  TASK_NAME="${TASK_NAME}_baseline"
  OPTIONS="$OPTIONS --baseline"
fi
echo $TASK_NAME

NAME="subject_model-${NETWORK}_preprocessing-${PREPROCESSING}_task-${TASK_NAME}_norm-${NORMALIZATION}_t-${T_BOOL}"

if [ $SPLITS > 0 ]; then
echo "Use of $SPLITS-fold cross validation, split $SPLIT"
NAME="${NAME}_splits-${SPLITS}"
fi

echo $NAME

# Run clinicaaddl
clinicadl train \
  subject \
  $CAPS_DIR \
  $TSV_PATH \
  $OUTPUT_DIR$NAME \
  $NETWORK \
  --diagnoses $TASK \
  --nproc $NUM_PROCESSORS \
  --batch_size $BATCH \
  --evaluation_steps $EVALUATION \
  --preprocessing $PREPROCESSING \
  --n_splits $SPLITS \
  --split $SPLIT \
  --accumulation_steps $ACCUMULATION \
  --epochs $EPOCHS \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --patience $PATIENCE \
  $OPTIONS
