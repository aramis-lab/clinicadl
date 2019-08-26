#!/usr/bin/env bash

#########################
#######  ADNI
#########################

## run baseline ADNI AD vs CN
python main_training.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --diagnosis_tsv_folder /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_diagnosis/train --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/baseline --baseline_or_longitudinal baseline --num_workers 72

## run longitudinal ADNI AD vs CN
python main_training.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --diagnosis_tsv_folder /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_diagnosis/train --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/longitudinal --baseline_or_longitudinal longitudinal --num_workers 72

## run baseline ADNI sMCI vs pMCI
python main_training.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --diagnosis_tsv_folder /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_diagnosis/train --source_classifer_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/baseline --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/SVM_results/transfer_learning/baseline --baseline_or_longitudinal baseline --num_workers 72

## run longitudinal ADNI sMCI vs pMCI
python main_training.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --diagnosis_tsv_folder /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_diagnosis/train --source_classifer_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/baseline --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/SVM_results/transfer_learning/longitudinal --baseline_or_longitudinal longitudinal --num_workers 72
