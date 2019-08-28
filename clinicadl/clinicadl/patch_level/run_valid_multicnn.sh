#!/usr/bin/env bash
#########################
#######  ADNI
#########################

## run baseline ADNI for task AD vs CN
for i in 0 1 2 3 4
do
    python main_test_multi_cnn.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/3d_patch/final_results/pytorch_AE_Conv_4_FC_2_bs32_lr_e5_only_finetuning_epoch20_ps_50_ss_50_baseline_all_patch_backup_multiCNN_MedIA --n_fold $i --mode valid
done

## run longitudinal ADNI for task AD vs CN
for i in 0 1 2 3 4
do
    python main_test_multi_cnn.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/3d_patch/final_results/pytorch_AE_Conv_4_FC_2_bs32_lr_e5_only_finetuning_epoch15_ps_50_ss_50_longitudinal_all_patch_multiCNN_MedIA --n_fold $i --mode valid
done

## run baseline ADNI for task sMCI vs pMCI
for i in 0 1 2 3 4
do
    python main_test_multi_cnn.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/sMCI_vs_pMCI_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/patch_level/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_baselineCNN_MedIA --n_fold $i --mode valid
done

## run longitudinal ADNI for task sMCI vs pMCI
for i in 0 1 2 3 4
do
    python main_test_multi_cnn.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/sMCI_vs_pMCI_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/patch_level/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_longCNN_MedIA --n_fold $i --mode valid
done