#!/bin/bash

set -x

SCRIPT="evaluation_singleCNN"
SELECTION="best_acc"

# Data management
COHORT=$1
DIAGNOSES="AD CN"

BATCH=16
GPU=1
NUM_WORKERS=8
OPTIONS="--hippocampus_roi"
WORKDIR="/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Zenodo_rerun/3_test"
OUTPUTDIR="patch3D_model-Conv4_FC3_preprocessing-linear_task-autoencoder_baseline-1_norm-1_hippocampus_splits-5"

TSVPATH="/network/lustre/iss01/home/elina.thibeausutre/data/Frontiers/$COHORT/lists_by_diagnosis/test/"
RESULTSPATH="$WORKDIR/test5/$OUTPUTDIR/"
IMGPATH="$WORKDIR/../../../${COHORT}"

if [ $COHORT = "ADNI" ]; then
IMGPATH="${IMGPATH}_rerun"
fi 

if [ $GPU = 1 ]; then
OPTIONS="${OPTIONS} --gpu"
fi

cd  /network/lustre/iss01/home/elina.thibeausutre/Code/AD-DL/clinicadl
python -m clinicadl.patch_level.$SCRIPT \
  $IMGPATH \
  $TSVPATH \
  $RESULTSPATH \
  --selection $SELECTION \
  --dataset "test-$COHORT"
  --diagnoses $DIAGNOSES
  --num_workers $NUM_WORKERS \
  $OPTIONS \
  --batch_size $BATCH \
  --prepare_dl
