#!/bin/bash
#SBATCH --partition=gpu_gct3
#SBATCH --time=20:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1        # on réserve des coeurs physiques et non logiques
#SBATCH --ntasks=1
#SBATCH --workdir=/gpfswork/rech/zft/upd53tc/jobs/AD-DL/train/patch_level/autoencoder
#SBATCH --output=./exp5/pytorch_job_%j.out
#SBATCH --error=./exp5/pytorch_job_%j.err
#SBATCH --job-name=3DAE_patch
#SBATCH --gres=gpu:1

#export http_proxy=http://10.10.2.1:8123
#export https_proxy=http://10.10.2.1:8123

# Experiment training autoencoder
eval "$(conda shell.bash hook)"
conda activate clinicadl_env_py37

# Network structure
NETWORK="Conv_4_FC_2"
COHORT="ADNI"
DATE="reproducibility_results"

# Input arguments to clinicadl
CAPS_DIR="$SCRATCH/../commun/datasets/$COHORT_rerun"
TSV_PATH="$HOME/code/AD-DL/data/$COHORT/lists_by_diagnosis/train"
OUTPUT_DIR="$SCRATCH/results/$DATE/"

# Computation ressources
NUM_PROCESSORS=8
GPU=1

# Dataset Management
PREPROCESSING='linear'
DIAGNOSES="AD CN MCI"
SPLITS=5
SPLIT=$1

# Training arguments
EPOCHS=20
BATCH=32
BASELINE=1
ACCUMULATION=2
EVALUATION=20
LR=1e-5
WEIGHT_DECAY=0
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


NAME="patch3D_model-${NETWORK}_preprocessing-${PREPROCESSING}_task-autoencoder_baseline-${BASELINE}_norm-${NORMALIZATION}"

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
  --patience $PATIENCE
  $OPTIONS
