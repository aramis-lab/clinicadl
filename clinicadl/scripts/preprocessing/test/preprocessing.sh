#!/bin/bash

BIDS_DIR=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/data/test_DL/BIDS/ADNI
CAPS_DIR=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/data/test_DL/CAPS/ADNI
TSV_FILE=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/data/test_DL/tsv/2_subjects_test.tsv
REF_TEMPLATE=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/data/test_DL/template/mni_icbm152_nlin_sym_09c_nifti/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii
WORKING_DIR=/network/lustre/dtlake01/aramis/users/mauricio.diazmelo/data/test_DL/working_dir


# Run clinicadl for preprocessing using 32 cores
clinicadl preprocessing --np 32 \
   $BIDS_DIR \
   $CAPS_DIR \
   $TSV_FILE \
   $REF_TEMPLATE \
   $WORKING_DIR
