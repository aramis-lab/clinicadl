#!/bin/bash

BIDS_DIR=/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/BIDS/ADNI_BIDS_T1_new/
CAPS_DIR=/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_rerun
TSV_FILE=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/tmp/DL/tsv_files/ADNI_after_qc.tsv
WORKING_DIR=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/tmp/DL/working_dir_postproc_rerun
REF_TEMPLATE=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/data/test_DL/template/mni_icbm152_nlin_sym_09c_nifti/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii


# Run clinicaaddl for preprocessing using 48 cores
clinicadl preprocessing --np 48 \
   $BIDS_DIR \
   $CAPS_DIR \
   $TSV_FILE \
   $REF_TEMPLATE \
   $WORKING_DIR
