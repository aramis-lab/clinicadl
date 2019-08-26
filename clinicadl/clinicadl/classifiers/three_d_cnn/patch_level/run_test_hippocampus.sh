#!/usr/bin/env bash


#########################
#######  ADNI
#########################

### run baseline ADNI for task AD vs CN
#for i in 0 1 2 3 4
#do
#    python main_test_hippocampus.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/ROI_based/Finished_exp/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch200_baseline_hippocampus50_with_es_MedIA --best_model_fold $i
#done

## run longitudinal ADNI for task AD vs CN
#for i in 0 1 2 3 4
#do
#    python main_test_hippocampus.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/ROI_based/Finished_exp/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_MedIA --best_model_fold $i
#done

## run baseline ADNI for task sMCI vs pMCI
#for i in 0 1 2 3 4
#do
#    python main_test_hippocampus.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/sMCI_vs_pMCI_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/ROI_based/transfered_from_CNN_AD_CN/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_baselineCNN_MedIA --best_model_fold $i
#done

## run longitudinal ADNI for task sMCI vs pMCI
#for i in 0 1 2 3 4
#do
#    python main_test_hippocampus.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/sMCI_vs_pMCI_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/ROI_based/transfered_from_CNN_AD_CN/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_longCNN_MedIA --best_model_fold $i
#done


##########################
########  AIBL
##########################

### run baseline AIBL for task AD vs CN
#for i in 0 1 2 3 4
#do
#    python main_test_hippocampus.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/AIBL --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/AIBL/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/ROI_based/Finished_exp/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch200_baseline_hippocampus50_with_es_MedIA --best_model_fold $i
#done
#
### run longitudinal AIBL for task AD vs CN
#for i in 0 1 2 3 4
#do
#    python main_test_hippocampus.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/AIBL --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/AIBL/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/ROI_based/Finished_exp/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_MedIA --best_model_fold $i
#done
#
### run baseline AIBL for task sMCI vs pMCI
#for i in 0 1 2 3 4
#do
#    python main_test_hippocampus.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/AIBL --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/AIBL/lists_by_task/test/sMCI_vs_pMCI_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/ROI_based/transfered_from_CNN_AD_CN/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_baselineCNN_MedIA --best_model_fold $i
#done
#
### run longitudinal AIBL for task sMCI vs pMCI
#for i in 0 1 2 3 4
#do
#    python main_test_hippocampus.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/AIBL --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/AIBL/lists_by_task/test/sMCI_vs_pMCI_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/ROI_based/transfered_from_CNN_AD_CN/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_longCNN_MedIA --best_model_fold $i
#done

#########################
#######  OASIS
#########################

## run baseline OASIS for task AD vs CN
#for i in 0 1 2 3 4
#do
#    python main_test_hippocampus.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/OASIS --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/OASIS/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/ROI_based/Finished_exp/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch200_baseline_hippocampus50_with_es_MedIA --best_model_fold $i
#done

## run longitudinal OASIS for task AD vs CN
#for i in 0 1 2 3 4
#do
#    python main_test_hippocampus.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/OASIS --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/OASIS/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/ROI_based/Finished_exp/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_MedIA --best_model_fold $i
#done
