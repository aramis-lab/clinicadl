#!/bin/bash
#SBATCH --partition=gpu_gct3
#SBATCH --time=20:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1        # on réserve des coeurs physiques et non logiques
#SBATCH --ntasks=1
#SBATCH --workdir=/gpfswork/rech/zft/upd53tc/jobs/AD-DL/train/subject_level/autoencoder
#SBATCH --output=./exp2/pytorch_job_%j.out
#SBATCH --error=./exp2/_pytorch_job_%j.err
#SBATCH --job-name=3DAE
#SBATCH --gres=gpu:1

#export http_proxy=http://10.10.2.1:8123
#export https_proxy=http://10.10.2.1:8123

# Experiment training autoencoder
eval "$(conda shell.bash hook)"
conda activate clinicadl_env_py37

# Network structure
NETWORK="Conv5_FC3"
COHORT="ADNI"
DATE="reproducibility_results"
CAPS_EXT="_skull_stripping"

# Input arguments to clinicadl
CAPS_DIR="$SCRATCH/../commun/datasets/$COHORT$CAPS_EXT"
TSV_PATH="$HOME/code/AD-DL/data/$COHORT/lists_by_diagnosis/train"
OUTPUT_DIR="$SCRATCH/results/$DATE/"

# Computation ressources
NUM_PROCESSORS=8
GPU=1

# Dataset Management
PREPROCESSING='mni'
DIAGNOSES="AD CN MCI"
BASELINE=1
SPLITS=5
SPLIT=$1

# Training arguments
EPOCHS=50
BATCH=6
ACCUMULATION=2
EVALUATION=20
LR=1e-4
GREEDY_LEARNING=0
SIGMOID=0
NORMALIZATION=1
PATIENCE=50

# Pretraining
T_BOOL=0
T_PATH=""
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
OPTIONS="$OPTIONS --pretrained_path $T_PATH -d $T_DIFF"
fi

if [ $BASELINE = 1 ]; then
echo "using only baseline data"
OPTIONS="$OPTIONS --baseline"
fi

NAME="model-${NETWORK}_preprocessing-${PREPROCESSING}_task-autoencoder_baseline-${BASELINE}_norm-${NORMALIZATION}"

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
  --train_autoencoder \
  --nproc $NUM_PROCESSORS \
  --batch_size $BATCH \
  --evaluation_steps $EVALUATION \
  --preprocessing $PREPROCESSING \
  --diagnoses $DIAGNOSES \
  --n_splits $SPLITS \
  --split $SPLIT \
  --accumulation_steps $ACCUMULATION \
  --epochs $EPOCHS \
  --learning_rate $LR \
  --patience $PATIENCE \
  $OPTIONS
