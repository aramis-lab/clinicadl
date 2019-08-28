#!/usr/bin/env bash


#########################
#######  ADNI
#########################

## run baseline ADNI
for i in 0 1 2 3 4
do
    python main_test.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/2d_slice/final_results/ResNet_fine_tune_last_block_top1fc_bs32_epoch50_lr6_wd4_es_15_baseline_MedIA --best_model_fold $i
done

## run longitudinal ADNI
for i in 0 1 2 3 4
do
    python main_test.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/2d_slice/final_results/ResNet_fine_tune_last_block_top1fc_bs32_epoch100_lr6_wd4_longitudinal_es15_MedIA --best_model_fold $i
done

## run baseline ADNI bad data split
for i in 0 1 2 3 4
do
    python main_test.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/2d_slice/final_results/ResNet_fine_tune_last_block_top1fc_bs32_epoch50_lr6_wd4_baseline_bad_data_split_MedIA --best_model_fold $i
done

##########################
########  AIBL
##########################

## run baseline AIBL
for i in 0 1 2 3 4
do
    python main_test.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/AIBL --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/AIBL/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/2d_slice/final_results/ResNet_fine_tune_last_block_top1fc_bs32_epoch50_lr6_wd4_es_15_baseline_MedIA --best_model_fold $i
done

## run longitudinal AIBL
for i in 0 1 2 3 4
do
    python main_test.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/AIBL --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/AIBL/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/2d_slice/final_results/ResNet_fine_tune_last_block_top1fc_bs32_epoch100_lr6_wd4_longitudinal_es15_MedIA --best_model_fold $i
done

## run baseline AIBL bad data split
for i in 0 1 2 3 4
do
    python main_test.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/AIBL --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/AIBL/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/2d_slice/final_results/ResNet_fine_tune_last_block_top1fc_bs32_epoch50_lr6_wd4_baseline_bad_data_split_MedIA --best_model_fold $i
done

#########################
#######  OASIS
#########################

## run baseline OASIS
for i in 0 1 2 3 4
do
    python main_test.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/OASIS --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/OASIS/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/2d_slice/final_results/ResNet_fine_tune_last_block_top1fc_bs32_epoch50_lr6_wd4_es_15_baseline_MedIA --best_model_fold $i
done

## run longitudinal OASIS
for i in 0 1 2 3 4
do
    python main_test.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/OASIS --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/OASIS/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/2d_slice/final_results/ResNet_fine_tune_last_block_top1fc_bs32_epoch100_lr6_wd4_longitudinal_es15_MedIA --best_model_fold $i
done

## run baseline OASIS bad data split
for i in 0 1 2 3 4
do
    python main_test.py --caps_directory /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/OASIS --diagnosis_tsv /network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/OASIS/lists_by_task/test/AD_vs_CN_baseline.tsv --output_dir /network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/2d_slice/final_results/ResNet_fine_tune_last_block_top1fc_bs32_epoch50_lr6_wd4_baseline_bad_data_split_MedIA --best_model_fold $i
done