#!/bin/bash
# Bash script to extract features used in classification algorithms proposed
# by clinicadl

CAPS_DIR=/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_rerun
TSV_FILE=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/tmp/DL/tsv_files/ADNI_after_qc.tsv
WORKING_DIR=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/tmp/DL/working_dir_postproc_rerun
METHOD='patch'

# Extract patches
# Run clinicadl for extract patches using 42 cores
clinicadl extract --np 42 \
   --patch_size 50 \
   --stride_size 50 \
   $CAPS_DIR \
   $TSV_FILE \
   $WORKING_DIR \
   $METHOD

# Extract slices
# Run clinicadl for extract sclices using 42 cores
METHOD='patch'
SLICE_MODE='rgb'

clinicadl extract --np 42 \
   --patch_size 50 \
   --stride_size 50 \
   --slice_mode $SLICE_MODE \
   $CAPS_DIR \
   $TSV_FILE \
   $WORKING_DIR \
   $METHOD
