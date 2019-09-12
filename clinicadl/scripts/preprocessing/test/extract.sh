#!/bin/bash
# Bash script for extract features used in classification algorithms proposed
# by clinicadl

CAPS_DIR=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/data/test_DL/CAPS/ADNI
TSV_FILE=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/data/test_DL/tsv/2_subjects_test.tsv
WORKING_DIR=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/data/test_DL/working_dir
METHOD='patch'

# Extract patches
# Run clinicadl for extract patches using 32 cores
clinicadl extract --np 32 \
   --patch_size 50 \
   --stride_size 50 \
   $CAPS_DIR \
   $TSV_FILE \
   $WORKING_DIR \
   $METHOD

# Extract slices
# Run clinicadl for extract sclices using 32 cores
METHOD='patch'
SLICE_MODE='rgb'

clinicadl extract --np 32 \
   --patch_size 50 \
   --stride_size 50 \
   --slice_mode $SLICE_MODE \
   $CAPS_DIR \
   $TSV_FILE \
   $WORKING_DIR \
   $METHOD
