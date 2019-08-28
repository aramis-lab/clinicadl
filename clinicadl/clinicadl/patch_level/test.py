import os
from classification_utils import load_model_test, extract_subject_name, evaluate_prediction
from model import Conv_4_FC_3
import pandas as pd



def multi_cnn_soft_majority_voting(model, output_dir, fi, num_cnn, mode='test'):
    """
    This is a function to do soft majority voting based on the num_cnn CNNs' performances
    :param output_dir:
    :param fi:
    :param num_cnn:
    :return:
    """

    # check the best validation acc for all the CNNs
    best_acc_cnns = []
    y_hat = []

    for n in range(num_cnn):
        # load the best trained model during the training
        _, _, _, best_predict = load_model_test(model, os.path.join(output_dir, 'best_model_dir',
                                                                                       "fold_" + str(fi), 'cnn-' + str(n), 'best_acc'),
                                                                          filename='model_best.pth.tar')
        best_acc_cnns.append(best_predict)

    ## delete the weak classifiers whose acc is smaller than 0.6
    weight_list = [0 if x < 0.7 else x for x in best_acc_cnns]
    weight_list = [x / sum(weight_list) for x in weight_list]
    if all(i == 0 for i in weight_list):
        raise Exception("The ensemble learning does not really work, all classifiers work bad!")

    ## read the validation patch-level results.
    for i in range(num_cnn):
        # load the best trained model during the training
        if i == 0:
            df = pd.io.parsers.read_csv(os.path.join(output_dir, 'performances', "fold_" + str(fi), 'cnn-' + str(i),
                                                     mode + '_patch_level_result-patch_index.tsv'), sep='\t')
            df_final = pd.DataFrame(columns=['subject', 'y', 'y_hat'])
            df_final['subject'] = df['subject'].apply(extract_subject_name)
            df_final['y'] = df['y']

        ##TODO, this is not correct, this is just the probability of the last epoch of validation
        tsv_path = os.path.join(output_dir, 'performances', "fold_" + str(fi), 'cnn-' + str(i), mode + '_patch_level_result-patch_index.tsv')
        proba_series = pd.io.parsers.read_csv(tsv_path, sep='\t')['probability']
        p0s = []
        p1s = []
        for j in range(len(proba_series)):
            p0 = weight_list[i] * eval(proba_series[j])[0]
            p1 = weight_list[i] * eval(proba_series[j])[1]
            p0s.append(p0)
            p1s.append(p1)
        p0s_series = pd.Series(p0s)
        p1s_series = pd.Series(p1s)

        ## adding the series into the final DataFrame
        ## insert the column of iteration
        df_final['cnn_' + str(i) + '_p0'] = p0s_series
        df_final['cnn_' + str(i) + '_p1'] = p1s_series

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

    for m in range(len(p0_soft)):
        proba_list = [p0_soft[m], p1_soft[m]]
        y_pred = proba_list.index(max(proba_list))
        y_hat.append(y_pred)

    ### convert y_hat list ot series then add into the dataframe
    y_hat_series = pd.Series(y_hat)
    p0_soft_series = pd.Series(p0_soft)
    p1_soft_series = pd.Series(p1_soft)
    df_final['y_hat'] = y_hat_series
    df_final['p0_soft'] = p0_soft_series
    df_final['p1_soft'] = p1_soft_series

    ## save the results into output_dir
    results_soft_tsv_path = os.path.join(output_dir, 'performances', "fold_" + str(fi),
                 'validation_subject_level_result_soft_vote_multi_cnn.tsv')
    df_final.to_csv(results_soft_tsv_path, index=False, sep='\t', encoding='utf-8')


    results = evaluate_prediction([int(e) for e in list(df_final.y)], [int(e) for e in list(
        df_final.y_hat)])  ## Note, y_hat here is not int, is string
    del results['confusion_matrix']

    metrics_soft_tsv_path = os.path.join(output_dir, 'performances', "fold_" + str(fi), mode + '_subject_level_metrics_soft_vote_multi_cnn.tsv')
    pd.DataFrame(results, index=[0]).to_csv(metrics_soft_tsv_path, index=False, sep='\t', encoding='utf-8')


model = eval('Conv_4_FC_3')()
output_dir = '/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/sMCI_pMCI/patch_level/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_longCNN'
num_cnn = 36
fi = 0
multi_cnn_soft_majority_voting(model, output_dir, fi, num_cnn, mode='validation')

