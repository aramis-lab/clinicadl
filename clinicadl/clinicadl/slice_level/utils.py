import torch
from torch.utils.data import Dataset
import os, shutil
from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from time import time
import tempfile

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


#################################
# CNN train / test
#################################

def train(model, data_loader, use_cuda, loss_func, optimizer, writer, epoch, model_mode="train",
          selection_threshold=None):
    """
    This is the function to train, validate or test the model, depending on the model_mode parameter.
    :param model:
    :param data_loader:
    :param use_cuda:
    :param loss_func:
    :param optimizer:
    :param writer:
    :param epoch:
    :return:
    """
    print("Start for %s!" % model_mode)
    global_step = None
    if model_mode == "train":
        columns = ['participant_id', 'session_id', 'patch_index', 'true_label', 'predicted_label', 'proba0', 'proba1']
        results_df = pd.DataFrame(columns=columns)
        total_loss = 0.0

        model.train()  # set the model to training mode
        print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))

        for i, data in enumerate(data_loader):
            # update the global steps
            global_step = i + epoch * len(data_loader)

            if use_cuda:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']

            gound_truth_list = labels.data.cpu().numpy().tolist()

            output = model(imgs)
            _, predicted = torch.max(output.data, 1)
            predict_list = predicted.data.cpu().numpy().tolist()
            batch_loss = loss_func(output, labels)
            total_loss += batch_loss.item()

            # calculate the batch balanced accuracy and loss
            batch_metrics = evaluate_prediction(gound_truth_list, predict_list)
            batch_accuracy = batch_metrics['balanced_accuracy']

            writer.add_scalar('classification accuracy', batch_accuracy, global_step)
            writer.add_scalar('loss', batch_loss.item(), global_step)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                row = [sub, data['session_id'][idx], data['patch_id'][idx],
                       labels[idx].item(), predicted[idx].item(),
                       output[idx, 0].item(), output[idx, 1]]
                row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
                results_df = pd.concat([results_df, row_df])

            # delete the temporary variables taking the GPU memory
            del imgs, labels, output, predicted, batch_loss, batch_accuracy
            torch.cuda.empty_cache()

        epoch_metrics = evaluate_prediction(results_df.true_label.values.astype(int),
                                            results_df.predicted_label.values.astype(int))
        accuracy_batch_mean = epoch_metrics['balanced_accuracy']
        loss_batch_mean = total_loss / len(data_loader)
        torch.cuda.empty_cache()

    elif model_mode == "valid":
        results_df, metrics_batch = test(model, data_loader, use_cuda, loss_func)

        # calculate the balanced accuracy
        _, metrics_subject = soft_voting(results_df, results_df, selection_threshold=selection_threshold)
        accuracy_batch_mean = metrics_subject['balanced_accuracy']
        total_loss = metrics_batch['total_loss']
        loss_batch_mean = total_loss / len(data_loader)

        writer.add_scalar('classification accuracy', accuracy_batch_mean, epoch)
        writer.add_scalar('loss', loss_batch_mean, epoch)

        torch.cuda.empty_cache()

    else:
        raise ValueError('This mode %s was not implemented. Please choose between train and valid' % model_mode)

    return results_df, accuracy_batch_mean, loss_batch_mean, global_step


def test(model, data_loader, use_cuda, loss_func):
    """
    The function to evaluate the testing data for the trained classifiers
    :param model:
    :param data_loader:
    :param use_cuda:
    :return:
    """

    columns = ['participant_id', 'session_id', 'slice_id', 'true_label', 'predicted_label', 'proba0', 'proba1']
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    print("Start evaluate the model!")
    if use_cuda:
        model.cuda()

    model.eval()  # set the model to evaluation mode
    torch.cuda.empty_cache()
    with torch.no_grad():
        print('The number of batches in this sampler based on the batch size: %s' % str(len(data_loader)))
        for i, data in enumerate(data_loader):
            if use_cuda:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']

            output = model(imgs)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                row = [sub, data['session_id'][idx], data['slice_id'][idx].item(),
                       labels[idx].item(), predicted[idx].item(),
                       output[idx, 0].item(), output[idx, 1].item()]

                row_df = pd.DataFrame(np.array(row).reshape(1, -1), columns=columns)
                results_df = pd.concat([results_df, row_df])

            del imgs, labels, output
            torch.cuda.empty_cache()

        # calculate the balanced accuracy
        results = evaluate_prediction(results_df.true_label.values.astype(int),
                                      results_df.predicted_label.values.astype(int))
        results_df.reset_index(inplace=True, drop=True)
        results['total_loss'] = total_loss
        torch.cuda.empty_cache()

    return results_df, results


def evaluate_prediction(y, y_hat):

    """
    This is a function to calculate the different metrics based on the list of true label and predicted label
    :param y: list
    :param y_hat: list
    :return:
    """

    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0

    tp = []
    tn = []
    fp = []
    fn = []

    for i in range(len(y)):
        if y[i] == 1:
            if y_hat[i] == 1:
                true_positive += 1
                tp.append(i)
            else:
                false_negative += 1
                fn.append(i)
        else:  # -1
            if y_hat[i] == 0:
                true_negative += 1
                tn.append(i)
            else:
                false_positive += 1
                fp.append(i)

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    if (true_positive + false_negative) != 0:
        sensitivity = true_positive / (true_positive + false_negative)
    else:
        sensitivity = 0.0

    if (false_positive + true_negative) != 0:
        specificity = true_negative / (false_positive + true_negative)
    else:
        specificity = 0.0

    if (true_positive + false_positive) != 0:
        ppv = true_positive / (true_positive + false_positive)
    else:
        ppv = 0.0

    if (true_negative + false_negative) != 0:
        npv = true_negative / (true_negative + false_negative)
    else:
        npv = 0.0

    balanced_accuracy = (sensitivity + specificity) / 2

    results = {'accuracy': accuracy,
               'balanced_accuracy': balanced_accuracy,
               'sensitivity': sensitivity,
               'specificity': specificity,
               'ppv': ppv,
               'npv': npv,
               'confusion_matrix': {'tp': len(tp), 'tn': len(tn), 'fp': len(fp), 'fn': len(fn)}
               }

    return results

# def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
#     """
#     This is the function to save the best model during validation process
#     :param state: the parameters that you wanna save
#     :param is_best: if the performance is better than before
#     :param checkpoint_dir:
#     :param filename:
#     :return:
#         checkpoint.pth.tar: this is the model trained by the last epoch, useful to retrain from this stopping point
#         model_best.pth.tar: if is_best is Ture, this is the best model during the validation, useful for testing the performances of the model
#     """
#     import shutil, os
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#     torch.save(state, os.path.join(checkpoint_dir, filename))
#     if is_best:
#         shutil.copyfile(os.path.join(checkpoint_dir, filename),  os.path.join(checkpoint_dir, 'model_best.pth.tar'))

# def subject_diagnosis_df(subject_session_df):
#     """
#     Creates a DataFrame with only one occurence of each subject and the most early diagnosis
#     Some subjects may not have the baseline diagnosis (ses-M00 doesn't exist)
#
#     :param subject_session_df: (DataFrame) a DataFrame with columns containing 'participant_id', 'session_id', 'diagnosis'
#     :return: DataFrame with the same columns as the input
#     """
#     temp_df = subject_session_df.set_index(['participant_id', 'session_id'])
#     subjects_df = pd.DataFrame(columns=subject_session_df.columns)
#     for subject, subject_df in temp_df.groupby(level=0):
#         session_nb_list = [int(session[5::]) for _, session in subject_df.index.values]
#         session_nb_list.sort()
#         session_baseline_nb = session_nb_list[0]
#         if session_baseline_nb < 10:
#             session_baseline = 'ses-M0' + str(session_baseline_nb)
#         else:
#             session_baseline = 'ses-M' + str(session_baseline_nb)
#         row_baseline = list(subject_df.loc[(subject, session_baseline)])
#         row_baseline.insert(0, subject)
#         row_baseline.insert(1, session_baseline)
#         row_baseline = np.array(row_baseline).reshape(1, len(row_baseline))
#         row_df = pd.DataFrame(row_baseline, columns=subject_session_df.columns)
#         subjects_df = subjects_df.append(row_df)
#
#     subjects_df.reset_index(inplace=True, drop=True)
#     return subjects_df


#################################
# Datasets
#################################

def load_split_by_slices(training_tsv, validation_tsv, mri_plane=0, val_size=0.15):
    """
    This is a function to gather the training and validation tsv together, then do the bad data split by slice.
    :param training_tsv:
    :param validation_tsv:
    :return:
    """

    df_training = pd.read_csv(training_tsv, sep='\t')
    df_validation = pd.read_csv(validation_tsv, sep='\t')
    df_all = pd.concat([df_training, df_validation])
    df_all = df_all.reset_index(drop=True)

    if mri_plane == 0:
        slices_per_patient = 169 - 40
        slice_index = range(20, 169 - 20)
    elif mri_plane == 1:
        slices_per_patient = 208 - 40
        slice_index = range(20, 208 - 20)
    else:
        slices_per_patient = 179 - 40
        slice_index = range(20, 179 - 20)

    participant_list = list(df_all['participant_id'])
    session_list = list(df_all['session_id'])
    label_list = list(df_all['diagnosis'])

    slice_participant_list = [ele for ele in participant_list for _ in range(slices_per_patient)]
    slice_session_list = [ele for ele in session_list for _ in range(slices_per_patient)]
    slice_label_list = [ele for ele in label_list for _ in range(slices_per_patient)]
    slice_index_list = slice_index * len(label_list)

    df_final = pd.DataFrame(columns=['participant_id', 'session_id', 'slice_id', 'diagnosis'])
    df_final['participant_id'] = np.array(slice_participant_list)
    df_final['session_id'] = np.array(slice_session_list)
    df_final['slice_id'] = np.array(slice_index_list)
    df_final['diagnosis'] = np.array(slice_label_list)

    y = np.array(slice_label_list)
    # split the train data into training and validation set
    skf_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=10000)
    indices = next(skf_2.split(np.zeros(len(y)), y))
    train_ind, valid_ind = indices

    df_sub_train = df_final.iloc[train_ind]
    df_sub_valid = df_final.iloc[valid_ind]

    train_tsv_path = os.path.join(tempfile.mkdtemp(), 'bad_data_split_train.tsv')
    valid_tsv_path = os.path.join(tempfile.mkdtemp(), 'bad_data_split_valid.tsv')

    df_sub_train.to_csv(train_tsv_path, sep='\t', index=False)
    df_sub_valid.to_csv(valid_tsv_path, sep='\t', index=False)

    return train_tsv_path, valid_tsv_path


class MRIDataset_slice(Dataset):
    """
    This class reads the CAPS of image processing pipeline of DL

    To note, this class processes the MRI to be RGB for transfer learning.

    Return: a Pytorch Dataset objective
    """

    def __init__(self, caps_directory, data_file, transformations=None, mri_plane=0):
        """
        Args:
            caps_directory (string): the output folder of image processing pipeline.
            transformations (callable, optional): if the data sample should be done some transformations or not, such as resize the image.

        To note, for each view:
            Axial_view = "[:, :, slice_i]"
            Coronal_veiw = "[:, slice_i, :]"
            Saggital_view= "[slice_i, :, :]"

        """
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1}
        self.mri_plane = mri_plane

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument datafile is not of correct type.')

        participant_list = list(self.df['participant_id'])
        session_list = list(self.df['session_id'])
        label_list = list(self.df['diagnosis'])

        # This dimension is for the output of image processing pipeline of Raw: 169 * 208 * 179
        if mri_plane == 0:
            self.slices_per_patient = 169 - 40
            self.slice_participant_list = [ele for ele in participant_list for _ in range(self.slices_per_patient)]
            self.slice_session_list = [ele for ele in session_list for _ in range(self.slices_per_patient)]
            self.slice_label_list = [ele for ele in label_list for _ in range(self.slices_per_patient)]
            self.slice_direction = 'sag'
        elif mri_plane == 1:
            self.slices_per_patient = 208 - 40
            self.slice_participant_list = [ele for ele in participant_list for _ in range(self.slices_per_patient)]
            self.slice_session_list = [ele for ele in session_list for _ in range(self.slices_per_patient)]
            self.slice_label_list = [ele for ele in label_list for _ in range(self.slices_per_patient)]
            self.slice_direction = 'cor'
        elif mri_plane == 2:
            self.slices_per_patient = 179 - 40
            self.slice_participant_list = [ele for ele in participant_list for _ in range(self.slices_per_patient)]
            self.slice_session_list = [ele for ele in session_list for _ in range(self.slices_per_patient)]
            self.slice_label_list = [ele for ele in label_list for _ in range(self.slices_per_patient)]
            self.slice_direction = 'axi'

    def __len__(self):
        return len(self.slice_participant_list)

    def __getitem__(self, idx):

        img_name = self.slice_participant_list[idx]
        sess_name = self.slice_session_list[idx]
        img_label = self.slice_label_list[idx]
        label = self.diagnosis_code[img_label]
        index_slice = idx % self.slices_per_patient

        # read the slices directly
        slice_path = os.path.join(self.caps_directory, 'subjects', img_name, sess_name, 't1',
                                  'preprocessing_dl',
                                  img_name + '_' + sess_name + '_space-MNI_res-1x1x1_axis-' +
                                  self.slice_direction + '_rgbslice-' + str(index_slice + 20) + '.pt')

        extracted_slice = torch.load(slice_path)
        extracted_slice = (extracted_slice - extracted_slice.min()) / (extracted_slice.max() - extracted_slice.min())

        # check if the slice has NAN value
        if torch.isnan(extracted_slice).any():
            print("Slice %s has nan values." % str(img_name + '_' + sess_name + '_' + str(index_slice + 20)))
            extracted_slice[torch.isnan(extracted_slice)] = 0

        if self.transformations:
            extracted_slice = self.transformations(extracted_slice)

        sample = {'image_id': img_name + '_' + sess_name + '_slice' + str(index_slice + 20), 'image': extracted_slice, 'label': label}

        return sample


class MRIDataset_slice_mixed(Dataset):
    """
    This class reads the CAPS of image processing pipeline of DL. However, this is used for the bad data split strategy

    To note, this class processes the MRI to be RGB for transfer learning.

    Return: a Pytorch Dataset objective
    """

    def __init__(self, caps_directory, data_file, transformations=None, mri_plane=0):
        """
        Args:
            caps_directory (string): the output folder of image processing pipeline.
            transformations (callable, optional): if the data sample should be done some transformations or not, such as resize the image.

        To note, for each view:
            Axial_view = "[:, :, slice_i]"
            Coronal_veiw = "[:, slice_i, :]"
            Saggital_view= "[slice_i, :, :]"

        """
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1}
        self.mri_plane = mri_plane

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument datafile is not of correct type.')

        self.participant_list = list(self.df['participant_id'])
        self.session_list = list(self.df['session_id'])
        self.slice_list = list(self.df['slice_id'])
        self.label_list = list(self.df['diagnosis'])

        if mri_plane == 0:
            self.slice_direction = 'sag'
        elif mri_plane == 1:
            self.slice_direction = 'cor'
        elif mri_plane == 2:
            self.slice_direction = 'axi'

    def __len__(self):
        return len(self.participant_list)

    def __getitem__(self, idx):

        img_name = self.participant_list[idx]
        sess_name = self.session_list[idx]
        slice_name = self.slice_list[idx]
        img_label = self.label_list[idx]
        label = self.diagnosis_code[img_label]

        slice_path = os.path.join(self.caps_directory, 'subjects', img_name, sess_name, 't1',
                                  'preprocessing_dl', img_name + '_' + sess_name + '_space-MNI_res-1x1x1_axis-' +
                                  self.slice_direction + '_rgbslice-' + str(slice_name) + '.pt')

        extracted_slice = torch.load(slice_path)
        extracted_slice = (extracted_slice - extracted_slice.min()) / (extracted_slice.max() - extracted_slice.min())

        # check if the slice has NAN value
        if torch.isnan(extracted_slice).any():
            print("Slice %s has nan values." % str(img_name + '_' + sess_name + '_' + str(slice_name)))
            extracted_slice[torch.isnan(extracted_slice)] = 0

        if self.transformations:
            extracted_slice = self.transformations(extracted_slice)

        sample = {'image_id': img_name + '_' + sess_name + '_slice' + str(slice_name), 'image': extracted_slice,
                  'label': label, 'participant_id': img_name, 'session_id': sess_name, 'slice_id': slice_name}

        return sample

# def load_model_from_log(model, optimizer, checkpoint_dir, filename='checkpoint.pth.tar'):
#     """
#     This is to load a saved model from the log folder
#     :param model:
#     :param checkpoint_dir:
#     :param filename:
#     :return:
#     """
#     from copy import deepcopy
#
#     ## set the model to be eval mode, we explicitly think that the model was saved in eval mode, otherwise, it will affects the BN and dropout
#
#     model.eval()
#     model_updated = deepcopy(model)
#     param_dict = torch.load(os.path.join(checkpoint_dir, filename))
#     model_updated.load_state_dict(param_dict['model'])
#     optimizer.load_state_dict(param_dict['optimizer'])
#
#     return model_updated, optimizer, param_dict['global_step'], param_dict['epoch']
#
# def load_model_test(model, checkpoint_dir, filename):
#     """
#     This is to load a saved model for testing
#     :param model:
#     :param checkpoint_dir:
#     :param filename:
#     :return:
#     """
#     from copy import deepcopy
#
#     ## set the model to be eval mode, we explicitly think that the model was saved in eval mode, otherwise, it will affects the BN and dropout
#     model.eval()
#     model_updated = deepcopy(model)
#     param_dict = torch.load(os.path.join(checkpoint_dir, filename))
#     model_updated.load_state_dict(param_dict['model'])
#
#     return model_updated, param_dict['global_step'], param_dict['epoch']
#


#################################
# Voting systems
#################################

def slice_level_to_tsvs(output_dir, results_df, results, fold, selection, dataset='train'):
    """
    Allows to save the outputs of the test function.

    :param output_dir: (str) path to the output directory.
    :param results_df: (DataFrame) the individual results per slice.
    :param results: (dict) the performances obtained on a series of metrics.
    :param fold: (int) the fold for which the performances were obtained.
    :param selection: (str) the metrics on which the model was selected (best_acc, best_loss)
    :param dataset: (str) the dataset on which the evaluation was performed.
    :return:
    """
    performance_dir = os.path.join(output_dir, 'performances', 'fold_%i' % fold, selection)

    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir)

    results_df.to_csv(os.path.join(performance_dir, dataset + '_slice_level_result-slice_index.tsv'), index=False,
                      sep='\t')

    del results['confusion_matrix']
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(performance_dir, dataset + '_slice_level_metrics.tsv'),
                                            index=False, sep='\t')


def retrieve_slice_level_results(output_dir, fold, selection, dataset):
    """Retrieve performance_df for single or multi-CNN framework."""
    result_tsv = os.path.join(output_dir, 'performances', 'fold_%i' % fold, selection,
                              dataset + '_slice_level_result-slice_index.tsv')
    performance_df = pd.read_csv(result_tsv, sep='\t')

    return performance_df


def soft_voting_to_tsvs(output_dir, fold, selection, dataset='test', selection_threshold=None):
    """
    This is for soft voting for subject-level performances
    :param performance_df: the pandas dataframe, including columns: iteration, y, y_hat, subject, probability
    :param selection: (str) the metrics on which the model was selected (best_acc, best_loss)

    ref: S. Raschka. Python Machine Learning., 2015
    :return:
    """

    # Choose which dataset is used to compute the weights of soft voting.
    if dataset in ['train', 'validation']:
        validation_dataset = dataset
    else:
        validation_dataset = 'validation'
    test_df = retrieve_slice_level_results(output_dir, fold, selection, dataset)
    validation_df = retrieve_slice_level_results(output_dir, fold, selection, validation_dataset)

    performance_path = os.path.join(output_dir, 'performances', 'fold_%i' % fold, selection)
    if not os.path.exists(performance_path):
        os.makedirs(performance_path)

    df_final, metrics = soft_voting(test_df, validation_df, selection_threshold=selection_threshold)

    df_final.to_csv(os.path.join(os.path.join(performance_path, dataset + '_subject_level_result_soft_vote.tsv')),
                    index=False, sep='\t')

    pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(output_dir, 'performances', 'fold_%i' % fold, selection,
                                                         dataset + '_subject_level_metrics_soft_vote.tsv'),
                                            index=False, sep='\t')


def soft_voting(performance_df, validation_df, selection_threshold=None):

    # Compute the slice accuracies on the validation set:
    right_classified_df = validation_df[validation_df['true_label'] == validation_df['predicted_label']]
    n_valid = len(validation_df.groupby(['participant_id', 'session_id']).nunique())
    slice_accuracies = right_classified_df['slice_id'].value_counts() / n_valid
    if selection_threshold is not None:
        slice_accuracies[slice_accuracies < selection_threshold] = 0
    weight_series = slice_accuracies / slice_accuracies.sum()

    # Add the weights to performance_df
    for idx in performance_df.index.values:
        slice_id = performance_df.loc[idx, 'slice_id']
        weight = weight_series.loc[slice_id]
        performance_df.loc[idx, 'weight'] = weight

    # do soft majority vote
    columns = ['participant_id', 'session_id', 'true_label', 'predicted_label']
    df_final = pd.DataFrame(columns=columns)
    for subject_session, subject_df in performance_df.groupby(['participant_id', 'session_id']):
        subject, session = subject_session
        num_slice = len(subject_df.predicted_label)
        p0_all = 0
        p1_all = 0
        # reindex the subject_df.probability
        proba0_series_reindex = subject_df.proba0.reset_index()
        proba1_series_reindex = subject_df.proba1.reset_index()
        weight_series_reindex = subject_df.weight.reset_index()
        y_series_reindex = subject_df.true_label.reset_index()
        y = y_series_reindex.true_label[0]

        for i in range(num_slice):

            p0 = weight_series_reindex.weight[i] * float(proba0_series_reindex.proba0[i])
            p1 = weight_series_reindex.weight[i] * float(proba1_series_reindex.proba1[i])

            p0_all += p0
            p1_all += p1

        proba_list = [p0_all, p1_all]
        y_hat = proba_list.index(max(proba_list))

        row_array = np.array(list([subject, session, y, y_hat])).reshape(1, 4)
        row_df = pd.DataFrame(row_array, columns=columns)
        df_final = df_final.append(row_df)

    results = evaluate_prediction(df_final.true_label.values.astype(int),
                                  df_final.predicted_label.values.astype(int))
    del results['confusion_matrix']

    return df_final, results
