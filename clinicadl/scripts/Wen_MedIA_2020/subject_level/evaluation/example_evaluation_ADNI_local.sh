#!/bin/bash

set -x

SCRIPT="evaluation_test.py"

# Data management
COHORT="ADNI"

BATCH=2
GPU=1
NUM_WORKERS=8
NUM_THREADS=0
OPTIONS=""
WORKDIR="/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Zenodo_rerun/test1"
OUTPUTDIR="subject_model-Conv5_FC3_preprocessing-linear_task-AD_CN_baseline_norm-0_t-0_splits-5"

TSVPATH="$HOME/code/AD-DL/data/$COHORT/lists_by_diagnosis/test/"
RESULTSPATH="$WORKDIR/3_test/$OUTPUTDIR/"
IMGPATH="$WORKDIR/../../${COHORT}_rerun"

if [ $GPU = 1 ]; then
OPTIONS="${OPTIONS} --gpu"
fi

python $HOME/code/AD-DL/clinicadl/clinicadl/subject_level/$SCRIPT \
  $RESULTSPATH \
  $IMGPATH \
  $TSVPATH \
  $COHORT \
  -w $NUM_WORKERS \
  --num_threads $NUM_THREADS $OPTIONS \
  --batch_size $BATCH
