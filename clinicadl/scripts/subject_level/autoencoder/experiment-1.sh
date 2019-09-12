#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --mem=17G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=.
#SBATCH --output=outputs/pytorch_job_%j.out
#SBATCH --error=outputs/pytorch_job_%j.err
#SBATCH --job-name=3DAE
#SBATCH --gres=gpu:1

#export http_proxy=http://10.10.2.1:8123
#export https_proxy=http://10.10.2.1:8123

# Experiment training autoencoder

# Network structure
NETWORK="Conv5_FC3"
COHORT='ADNI'

# Input arguments to clinicadl
CAPS_DIR="/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_rerun"
TSV_PATH="/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/code/AD-DL/data/$COHORT/lists_by_diagnosis/train"
OUTPUT_DIR="/network/lustre/iss01/home/mauricio.diazmelo/ADNI_rerun/results/$DATE/"

# Computation ressources
NUM_PROCESSORS=8

# Dataset Management
PREPROCESSING='linear'
DIAGNOSES="AD CN MCI"
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


OPTIONS=""
DATE="reproducibility_results"

NAME="model-${NETWORK}_preprocessing-${PREPROCESSING}_task-autoencoder_baseline-${BASELINE}_norm-${NORMALIZATION}"

if [ $SPLITS > 0 ]; then
echo "Use of $SPLITS-fold cross validation, split $SPLIT"
NAME="${NAME}_splits-${SPLITS}"
fi

echo $NAME
echo $OPTIONS

# Run clinicadl
clinicadl train \
  $CAPS_DIR \
  $TSV_PATH \
  $OUTPUT_DIR$NAME \
  $NETWORK \
  --train_autoencoder \  
  --use-gpu \
  --nproc $NUM_PROCESSORS \
  --batch_size $BATCH \
  --evaluation_steps $EVALUATION \
  --preprocessing $PREPROCESSING \
  --diagnoses $DIAGNOSES \
  --baseline \
  --minmaxnormalization \
  --n_splits $NSPLITS \
  --split $SPLIT \
  --accumulation_steps $ACCUMULATION \
  --eprochs $EPOCHS \
  --learning_rate $LR
  --patience $PATIENCE
  --tolerance $TOLERANCE
