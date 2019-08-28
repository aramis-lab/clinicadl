import os
from classification_utils import load_model_test, extract_subject_name, evaluate_prediction
from model import Conv_4_FC_3
import pandas as pd
import numpy as np

def multi_cnn_soft_majority_voting(output_dir, n_fold, num_cnn, mode='test'):
    """
    This is a function to do soft majority voting based on the num_cnn CNNs' performances
    :param output_dir:
    :param fi: the i-th fold
    :param num_cnn:
    :return:
    """

    for fi in range(n_fold):
        # check the best test patch-level acc for all the CNNs
        best_acc_cnns = []
        y_hat = []

        for n in range(num_cnn):
            # load the patch-level balanced accuracy from the tsv files
            tsv_path_metric = os.path.join(output_dir, 'performances', "fold_" + str(fi), 'cnn-' + str(n), mode + '_patch_level_metrics.tsv')

            best_ba = pd.io.parsers.read_csv(tsv_path_metric, sep='\t')['balanced_accuracy']

            best_acc_cnns.append(best_ba[0])

        ## delete the weak classifiers whose acc is smaller than 0.6
        ba_list = [0 if x < 0.7 else x for x in best_acc_cnns]
        if all(ba == 0 for ba in ba_list):
            print("Pay attention, all the CNNs did not perform well for %d -th fold" % (fi))
        else:

            weight_list = [x / sum(ba_list) for x in ba_list]

            ## read the test data patch-level probability results.
            for i in range(num_cnn):
                # load the best trained model during the training

                df = pd.io.parsers.read_csv(os.path.join(output_dir, 'performances', "fold_" + str(fi), 'cnn-' + str(i),
                                                        mode + '_patch_level_result-patch_index.tsv'), sep='\t')
                if i == 0:
                    df_final = pd.DataFrame(columns=['subject', 'y', 'y_hat'])
                    df_final['subject'] = df['subject'].apply(extract_subject_name)
                    df_final['y'] = df['y']

                proba_series = df['probability']
                p0s = []
                p1s = []
                for j in range(len(proba_series)):
                    p0 = weight_list[i] * eval(proba_series[j])[0]
                    p1 = weight_list[i] * eval(proba_series[j])[1]
                    p0s.append(p0)
                    p1s.append(p1)
                p0s_array = np.asarray(p0s)
                p1s_array = np.asarray(p1s)

                ## adding the series into the final DataFrame
                ## insert the column of iteration
                df_final['cnn_' + str(i) + '_p0'] = p0s_array
                df_final['cnn_' + str(i) + '_p1'] = p1s_array

            ## based on the p0 and p1 from all the CNNs, calculate the y_hat
            p0_final = []
            p1_final = []
            for k in range(num_cnn):
                p0_final.append(df_final['cnn_' + str(k) + '_p0'].tolist())
            for k in range(num_cnn):
                p1_final.append(df_final['cnn_' + str(k) + '_p1'].tolist())

            ## element-wise adding to calcuate the final probability
            p0_soft = [sum(x) for x in zip(*p0_final)]
            p1_soft = [sum(x) for x in zip(*p1_final)]

            ## adding the final p0 and p1 to the dataframe
            df_final['p0'] = np.asarray(p0_soft)
            df_final['p1'] = np.asarray(p1_soft)

            for m in range(len(p0_soft)):
                proba_list = [p0_soft[m], p1_soft[m]]
                y_pred = proba_list.index(max(proba_list))
                y_hat.append(y_pred)

            ## adding y_pred into the dataframe
            df_final['y_hat'] = np.asarray(y_hat)

            ## save the results into output_dir
            results_soft_tsv_path = os.path.join(output_dir, 'performances', "fold_" + str(fi),
                         mode + '_subject_level_result_soft_vote_multi_cnn.tsv')
            df_final.to_csv(results_soft_tsv_path, index=False, sep='\t', encoding='utf-8')


            results = evaluate_prediction([int(e) for e in list(df_final.y)], [int(e) for e in list(
                df_final.y_hat)])  ## Note, y_hat here is not int, is string
            del results['confusion_matrix']

            metrics_soft_tsv_path = os.path.join(output_dir, 'performances', "fold_" + str(fi), mode + '_subject_level_metrics_soft_vote_multi_cnn.tsv')
            pd.DataFrame(results, index=[0]).to_csv(metrics_soft_tsv_path, index=False, sep='\t', encoding='utf-8')


#############ADNI

## for baseline AD vs CN ADNI
# output_dir = '/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/3d_patch/final_results/pytorch_AE_Conv_4_FC_2_bs32_lr_e5_only_finetuning_epoch20_ps_50_ss_50_baseline_all_patch_backup_multiCNN_MedIA'

## for longitudinal AD vs CN ADNI
# output_dir = '/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/3d_patch/final_results/pytorch_AE_Conv_4_FC_2_bs32_lr_e5_only_finetuning_epoch15_ps_50_ss_50_longitudinal_all_patch_multiCNN_MedIA'

## for baseline sMCI vs pMCI ADNI
# output_dir = '/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/patch_level/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_baselineCNN_MedIA'

## for longitudinal sMCI vs pMCI ADNI
output_dir = '/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/patch_level/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_longCNN_MedIA'



#############AIBL
## for baseline AD vs CN AIBL
# output_dir = '/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/3d_patch/final_results/pytorch_AE_Conv_4_FC_2_bs32_lr_e5_only_finetuning_epoch20_ps_50_ss_50_baseline_all_patch_backup_multiCNN_MedIA'

## for longitudinal AD vs CN AIBL
# output_dir = '/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/3d_patch/final_results/pytorch_AE_Conv_4_FC_2_bs32_lr_e5_only_finetuning_epoch15_ps_50_ss_50_longitudinal_all_patch_multiCNN_MedIA'

## for baseline sMCI vs pMCI AIBL
# output_dir = '/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/patch_level/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_baselineCNN_MedIA'

## for longitudinal sMCI vs pMCI AIBL
# output_dir = '/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/patch_level/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_longCNN_MedIA'


#############OASIS
## for baseline AD vs CN OASIS
# output_dir = '/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/3d_patch/final_results/pytorch_AE_Conv_4_FC_2_bs32_lr_e5_only_finetuning_epoch20_ps_50_ss_50_baseline_all_patch_backup_multiCNN_MedIA'

## for longitudinal AD vs CN OASIS
# output_dir = '/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/3d_patch/final_results/pytorch_AE_Conv_4_FC_2_bs32_lr_e5_only_finetuning_epoch15_ps_50_ss_50_longitudinal_all_patch_multiCNN_MedIA'


num_cnn = 36
n_fold = 5
multi_cnn_soft_majority_voting(output_dir, n_fold, num_cnn, mode='test')

