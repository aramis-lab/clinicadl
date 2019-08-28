#!/usr/bin/env bash


#########################
#######  ADNI
#########################

## run baseline ADNI AD vs CN
#python main_test.py --caps_directory_source /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --caps_directory_target /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --diagnosis_tsv_test /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/baseline --group_id_target ADNIbl --group_id_source ADNIbl

## run longitudinal ADNI AD vs CN
python main_test.py --caps_directory_source /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --caps_directory_target /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --diagnosis_tsv_test /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/longitudinal --group_id_target ADNIbl --group_id_source ADNIbl

## run baseline ADNI sMCI vs pMCI
python main_test.py --caps_directory_source /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --caps_directory_target /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --diagnosis_tsv_test /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_task/test/sMCI_vs_pMCI_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/SVM_results/transfer_learning/baseline --group_id_target ADNIbl --group_id_source ADNIbl

## run longitudinal ADNI sMCI vs pMCI
python main_test.py --caps_directory_source /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --caps_directory_target /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --diagnosis_tsv_test /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_task/test/sMCI_vs_pMCI_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/SVM_results/transfer_learning/longitudinal --group_id_target ADNIbl --group_id_source ADNIbl


##########################
########  AIBL
##########################

## run baseline ADNI AD vs CN
python main_test.py --caps_directory_source /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --caps_directory_target /network/lustre/dtlake01/aramis/datasets/aibl/caps --diagnosis_tsv_test /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/AIBL/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir_source /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/baseline --output_dir_target /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/baseline --group_id_target AIBLbl --group_id_source ADNIbl

## run longitudinal ADNI AD vs CN
python main_test.py --caps_directory_source /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --caps_directory_target /network/lustre/dtlake01/aramis/datasets/aibl/caps --diagnosis_tsv_test /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/AIBL/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/longitudinal --group_id_target AIBLbl --group_id_source ADNIbl

## run baseline ADNI sMCI vs pMCI
python main_test.py --caps_directory_source /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --caps_directory_target /network/lustre/dtlake01/aramis/datasets/aibl/caps --diagnosis_tsv_test /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/AIBL/lists_by_task/test/sMCI_vs_pMCI_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/SVM_results/transfer_learning/baseline --group_id_target AIBLbl --group_id_source ADNIbl

## run longitudinal ADNI sMCI vs pMCI
python main_test.py --caps_directory_source /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm --caps_directory_target /network/lustre/dtlake01/aramis/datasets/aibl/caps --diagnosis_tsv_test /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/AIBL/lists_by_task/test/sMCI_vs_pMCI_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/SVM_results/transfer_learning/longitudinal --group_id_target AIBLbl --group_id_source ADNIbl


#########################
#######  OASIS
#########################

