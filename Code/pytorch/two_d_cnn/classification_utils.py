import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nibabel as nib
from torch.utils.data import Dataset
import random
from clinica.pipelines.machine_learning.svm_utils import evaluate_prediction
import os, shutil
from skimage.transform import resize
from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def train(model, data_loader, use_cuda, loss_func, optimizer, writer, epoch_i, train_mode="train"):
    """
    This is the function to train, validate or test the model, depending on the train_mode parameter.
    :param model:
    :param data_loader:
    :param use_cuda:
    :param loss_func:
    :param optimizer:
    :param writer:
    :param epoch_i:
    :return:
    """
    # main training loop
    correct_cnt = 0.0
    acc = 0.0
    subjects = []
    y_ground = []
    y_hat = []
    if train_mode == "train":
        model.train() ## set the model to training mode
    else:
        model.eval() ## set the model to evaluation mode

    for i, subject_data in enumerate(data_loader):
        # for each iteration, the train data contains batch_size * n_slices_in_each_subject images
        loss_batch = 0.0
        acc_batch = 0.0
        num_slice = len(subject_data)
        for j in range(num_slice):
            data_dic = subject_data[j]
            if use_cuda:
                imgs, labels = Variable(data_dic['image']).cuda(), Variable(data_dic['label']).cuda()
            else:
                imgs, labels = Variable(data_dic['image']), Variable(data_dic['label'])

            ## add the participant_id + session_id
            image_ids = data_dic['image_id']
            subjects.extend(image_ids)

            # TO track of indices, int64 is a better choice for large models.
            integer_encoded = labels.data.cpu().numpy()
            gound_truth_list = integer_encoded.tolist()
            y_ground.extend(gound_truth_list)
            ground_truth = Variable(torch.from_numpy(integer_encoded)).long()

            print 'The group true label is %s' % str(labels)
            if use_cuda:
                ground_truth = ground_truth.cuda()
            output = model(imgs)
            _, predict = output.topk(1)
            predict_list = predict.data.cpu().numpy().tolist()
            y_hat.extend([item for sublist in predict_list for item in sublist])
            if train_mode == "train" or train_mode == 'valid':
                loss = loss_func(output, ground_truth)
                loss_batch += loss
            correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
            correct_cnt += correct_this_batch
            # To monitor the training process using tensorboard, we only display the training loss and accuracy, the other performance metrics, such
            # as balanced accuracy, will be saved in the tsv file.
            accuracy = float(correct_this_batch) / len(ground_truth)
            acc_batch += accuracy
            if train_mode == "train":
                print ("For batch %d slice %d training loss is : %f") % (i, j, loss.item())
                print ("For batch %d slice %d training accuracy is : %f") % (i, j, accuracy)
            elif train_mode == "valid":
                print ("For batch %d slice %d validation accuracy is : %f") % (i, j, accuracy)
                print ("For batch %d slice %d validation loss is : %f") % (i, j, loss.item())
            elif train_mode == "test":
                print ("For batch %d slice %d validate accuracy is : %f") % (i, j, accuracy)

            # Unlike tensorflow, in Pytorch, we need to manully zero the graident before each backpropagation step, becase Pytorch accumulates the gradients
            # on subsequent backward passes. The initial designing for this is convenient for training RNNs.
            if train_mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if train_mode == "train" or "valid":
            writer.add_scalar('slice-level accuracy', acc_batch / num_slice, i + epoch_i * len(data_loader.dataset))
        else:
            writer.add_scalar('test_accuracy_subject', acc_batch / num_slice, i)
        ##for sanity check that we do not do bad things for preparing the data
        if train_mode == "train":
            writer.add_scalar('loss', loss_batch / num_slice, i + epoch_i * len(data_loader.dataset))
            writer.add_image('example_image', imgs.int(), i + epoch_i * len(data_loader.dataset))
        ## add all accuracy for each iteration
        acc += acc_batch / num_slice

    acc_mean = acc / len(data_loader)

    return imgs, subjects, y_ground, y_hat, acc_mean

# def validate(model, valid_loader, use_cuda, loss_func, writer_valid, epoch_i):
#     """
#     This is the function to validate the CNN with validation data
#     :param model:
#     :param valid_loader:
#     :param use_cuda:
#     :param loss_func:
#     :param writer:
#     :param epoch_i:
#     :return:
#     """
#     correct_cnt = 0
#     acc = 0.0
#     model.eval()
#     for i, valid_data in enumerate(valid_loader):
#         loss_batch = 0.0
#         acc_batch = 0.0
#         for j in range(len(valid_data)):
#             data_dic = valid_data[j]
#             if use_cuda:
#                 imgs, labels = Variable(data_dic['image'], volatile=True).cuda(), Variable(data_dic['label'],
#                                                                                            volatile=True).cuda()
#             else:
#                 imgs, labels = Variable(data_dic['image'], volatile=True), Variable(data_dic['label'],
#                                                                         volatile=True)
#             integer_encoded = labels.data.cpu().numpy()
#             ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
#             print 'The group true label is %s' % str(labels)
#             if use_cuda:
#                 ground_truth = ground_truth.cuda()
#             valid_output = model(imgs)
#             _, predict = valid_output.topk(1)
#             loss = loss_func(valid_output, ground_truth)
#             loss_batch += loss
#             correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
#             correct_cnt += correct_this_batch
#             accuracy = float(correct_this_batch) / len(ground_truth)
#             acc_batch += accuracy
#
#             print ("For batch %d slice %d validation loss is : %f") % (i, j, loss.item())
#             print ("For batch %d slice %d validation accuracy is : %f") % (i, j, accuracy)
#
#         writer_valid.add_scalar('validation_accuracy', acc_batch / len(valid_data), i + epoch_i * len(valid_loader.dataset))
#         writer_valid.add_scalar('validation_loss', loss_batch / len(valid_data), i + epoch_i * len(valid_loader.dataset))
#
#         acc += acc_batch / len(valid_data)
#
#     acc_mean = acc / len(valid_loader)
#
#     return acc_mean
#
# def test(model, test_loader, use_cuda, writer_test):
#     """
#     This is the function to test the CNN with testing data
#     :param model:
#     :param test_loader:
#     :param use_cuda:
#     :param writer:
#     :return:
#     """
#     correct_cnt = 0
#     model.eval()
#     for i, test_data in enumerate(test_loader):
#         vote = []
#         acc_batch = 0.0
#
#         for j in range(len(test_data)):
#             data_dic = test_data[j]
#             if use_cuda:
#                 imgs, labels = Variable(data_dic['image'], volatile=True).cuda(), Variable(data_dic['label'],
#                                                                                            volatile=True).cuda()
#             else:
#                 imgs, labels = Variable(data_dic['image'], volatile=True), Variable(data_dic['label'],
#                                                                                     volatile=True)
#
#             ## fpr slice-level accuracy
#             integer_encoded = labels.data.cpu().numpy()
#             # target should be LongTensor in loss function
#             ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
#             print 'The group true label is %s' % str(labels)
#             if use_cuda:
#                 ground_truth = ground_truth.cuda()
#             test_output = model(imgs)
#             _, predict = test_output.topk(1)
#             vote.append(predict)
#             correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
#             accuracy_slice = float(correct_this_batch) / len(ground_truth)
#             acc_batch += accuracy_slice
#
#             print ("For batch %d slice %d test accuracy is : %f") % (i, j, accuracy_slice)
#
#         writer_test.add_scalar('test_accuracy_slice', acc_batch / len(test_data), i)
#
#         ## for subject-level accuracy
#         vote = torch.cat(vote, 1)
#         final_vote, _ = torch.mode(vote, 1) ## This is the majority vote for each subject, based on all the slice-level results
#         ground_truth = test_data[0]['label']
#         correct_this_batch_subject = (final_vote.cpu().data == ground_truth).sum()
#         accuracy_subject = float(correct_this_batch_subject) / len(ground_truth)
#
#         print("Subject level for batch %d testing accuracy is : %f") % (i, accuracy_subject)
#         writer_test.add_scalar('test_accuracy_subject', accuracy_subject, i)
#
#     return accuracy_subject

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    This is the function to save the best model during validation process
    :param state: the parameters that you wanna save
    :param is_best: if the performance is better than before
    :param checkpoint_dir:
    :param filename:
    :return:
    """
    import shutil, os
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_dir, filename),  os.path.join(checkpoint_dir, 'model_best.pth.tar'))

def subject_diagnosis_df(subject_session_df):
    """
    Creates a DataFrame with only one occurence of each subject and the most early diagnosis
    Some subjects may not have the baseline diagnosis (ses-M00 doesn't exist)

    :param subject_session_df: (DataFrame) a DataFrame with columns containing 'participant_id', 'session_id', 'diagnosis'
    :return: DataFrame with the same columns as the input
    """
    temp_df = subject_session_df.set_index(['participant_id', 'session_id'])
    subjects_df = pd.DataFrame(columns=subject_session_df.columns)
    for subject, subject_df in temp_df.groupby(level=0):
        session_nb_list = [int(session[5::]) for _, session in subject_df.index.values]
        session_nb_list.sort()
        session_baseline_nb = session_nb_list[0]
        if session_baseline_nb < 10:
            session_baseline = 'ses-M0' + str(session_baseline_nb)
        else:
            session_baseline = 'ses-M' + str(session_baseline_nb)
        row_baseline = list(subject_df.loc[(subject, session_baseline)])
        row_baseline.insert(0, subject)
        row_baseline.insert(1, session_baseline)
        row_baseline = np.array(row_baseline).reshape(1, len(row_baseline))
        row_df = pd.DataFrame(row_baseline, columns=subject_session_df.columns)
        subjects_df = subjects_df.append(row_df)

    subjects_df.reset_index(inplace=True, drop=True)
    return subjects_df


def multiple_time_points(df, subset_df):
    """
    Returns a DataFrame with all the time points of each subject

    :param df: (DataFrame) the reference containing all the time points of all subjects.
    :param subset_df: (DataFrame) the DataFrame containing the subset of subjects.
    :return: mtp_df (DataFrame) a DataFrame with the time points of the subjects of subset_df
    """

    mtp_df = pd.DataFrame(columns=df.columns)
    temp_df = df.set_index('participant_id')
    for idx in subset_df.index.values:
        subject = subset_df.loc[idx, 'participant_id']
        subject_df = temp_df.loc[subject]
        if isinstance(subject_df, pd.Series):
            subject_id = subject_df.name
            row = list(subject_df.values)
            row.insert(0, subject_id)
            subject_df = pd.DataFrame(np.array(row).reshape(1, len(row)), columns=df.columns)
            mtp_df = mtp_df.append(subject_df)
        else:
            mtp_df = mtp_df.append(subject_df.reset_index())

    mtp_df.reset_index(inplace=True, drop=True)
    return mtp_df


def split_subjects_to_tsv(diagnoses_tsv, n_splits=5, val_size=0.15):
    """
    Write the tsv files corresponding to the train/val/test splits of all folds

    :param diagnoses_tsv: (str) path to the tsv file with diagnoses
    :param n_splits: (int) the number of splits wanted in the cross-validation
    :param val_size: (float) proportion of the train set being used for validation
    :return: None
    """

    df = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')
    if 'diagnosis' not in list(df.columns.values):
        raise Exception('Diagnoses file is not in the correct format.')
    # check if we have duplicated row in the tsv, this will cause problem if yes
    if df.duplicated().unique().size > 1:
        raise ValueError("There are duplicated rows in the tsv files, please double check it!!!")
    # Here we reduce the DataFrame to have only one diagnosis per subject (multiple time points case)
    diagnosis_df = subject_diagnosis_df(df)

    diagnoses_list = list(diagnosis_df.diagnosis)
    unique = list(set(diagnoses_list))
    y = np.array([unique.index(x) for x in diagnoses_list])  # There is one label per diagnosis depending on the order

    splits = StratifiedKFold(n_splits=n_splits, shuffle=True)

    n_iteration = 0
    for train_index, test_index in splits.split(np.zeros(len(y)), y):

        y_train = y[train_index]
        diagnosis_df_train = diagnosis_df.loc[train_index]

        # split the train data into training and validation set
        skf_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
        indices = next(skf_2.split(np.zeros(len(y_train)), y_train))
        train_ind, valid_ind = indices

        df_test = diagnosis_df.iloc[test_index]
        df_sub_valid = diagnosis_df_train.iloc[valid_ind]
        df_sub_train = diagnosis_df_train.iloc[train_ind]
        df_valid = multiple_time_points(df, df_sub_valid)
        df_train = multiple_time_points(df, df_sub_train)
        df_test = multiple_time_points(df, df_test)

        df_train.to_csv(path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(n_iteration) + '_splits-' + str(n_splits) + '_valid_size-' + str(val_size) + '_train.tsv'), sep='\t', index=False)
        df_test.to_csv(path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(n_iteration) + '_splits-' + str(n_splits) + '_valid_size-' + str(val_size) + '_test.tsv'), sep='\t', index=False)
        df_valid.to_csv(path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(n_iteration) + '_splits-' + str(n_splits) + '_valid_size-' + str(val_size) + '_valid.tsv'), sep='\t', index=False)
        n_iteration += 1


def load_split(diagnoses_tsv, fold, n_splits, val_size=0.15):
    """
    Loads the

    :param diagnoses_tsv: (str) path to the tsv file with diagnoses
    :param fold: (int) the number of the current fold
    :return: 3 DataFrame
        training_tsv
        test_tsv
        valid_tsv
    """
    training_tsv = path.join(path.dirname(diagnoses_tsv),
                             path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(fold) + '_splits-' + str(n_splits) + '_valid_size-' + str(val_size) + '_train.tsv')
    test_tsv = path.join(path.dirname(diagnoses_tsv),
                         path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(fold) + '_splits-' + str(n_splits) + '_valid_size-' + str(val_size) + '_test.tsv')
    valid_tsv = path.join(path.dirname(diagnoses_tsv),
                          path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(fold) + '_splits-' + str(n_splits) + '_valid_size-' + str(val_size) + '_valid.tsv')
    return training_tsv, test_tsv, valid_tsv

def check_and_clean(d):

  if os.path.exists(d):
      shutil.rmtree(d)
  os.mkdir(d)

class mri_to_rgb_transfer(Dataset):
    """
    This class reads the CAPS of image processing pipeline of DL

    To note, this class processes the MRI to be RGB for transfer learning.

    Return: a Pytorch Dataset objective
    """

    def __init__(self, caps_directory, tsv, transform=None):
        """
        Args:
            caps_directory (string): the output folder of image processing pipeline.
            tsv (string): the tsv containing three columns, participant_id, session_id and diagnosis.
            transform (callable, optional): if the data sample should be done some transform or not, such as resize the image.

        To note, for each view:
            Axial_view = "[:, :, slice_i]"
            Coronal_veiw = "[:, slice_i, :]"
            Saggital_view= "[slice_i, :, :]"

        """
        self.caps_directory = caps_directory
        self.tsv = tsv
        self.transform = transform

        df = pd.io.parsers.read_csv(tsv, sep='\t')
        if ('diagnosis' != list(df.columns.values)[2]) and ('session_id' != list(df.columns.values)[1]) and (
            'participant_id' != list(df.columns.values)[0]):
            raise Exception('the data file is not in the correct format.')
        participant_list = list(df['participant_id'])
        session_list = list(df['session_id'])
        label_list = list(df['diagnosis'])

        self.participant_list = participant_list
        self.session_list = session_list
        self.label_list = label_list

    def __len__(self):
        return len(self.participant_list)

    def __getitem__(self, idx):

        img_name = self.participant_list[idx]
        img_label = self.label_list[idx]
        sess_name = self.session_list[idx]
        image_path = os.path.join(self.caps_directory, 'subjects', img_name, sess_name, 't1', 'preprocessing_dl', img_name + '_' + sess_name + '_space-MNI_res-1x1x1.nii.gz')
        samples = []
        if img_label == 'CN':
            label = 0
        elif img_label == 'AD':
            label = 1
        elif img_label == 'MCI':
            label = 2
        elif img_label == 'pMCI':
            label = 3
        elif img_label == 'sMCI':
            label = 4
        else:
            raise ValueError('The label you specified is not correct, please double check it!')

        ### get all the slices from the three view, basically, we take 3 slices and create a RBG image so that we can using this for transferring learning.
        ## For axial view
        axial_image_list = slices_to_rgb(image_path, 2, rgb_mode='single_slice')
        ## For coronal view
        coronal_image_list = slices_to_rgb(image_path, 1, rgb_mode='single_slice')
        ## For saggital view
        saggital_image_list = slices_to_rgb(image_path, 0, rgb_mode='single_slice')

        for img_rgb in (axial_image_list, coronal_image_list, saggital_image_list):
            for img in img_rgb:
                if self.transform:
                    img = self.transform(img)
                sample = {'image_id': img_name + '_' + sess_name, 'image': img, 'label': label}
                samples.append(sample)
        random.shuffle(samples)
        return samples


def slices_to_rgb(image_path, view, rgb_mode='single_slice'):
    """
    This is a function to grab each slice in each view and create a rgb image for transferring learning: duplicate the slices into R, G, B channel
    :param image_path:
    :param view:
    :return:

    To note, for each view:
    Axial_view = "[:, :, slice_i]"
    Coronal_veiw = "[:, slice_i, :]"
    Saggital_view= "[slice_i, :, :]"
    """


    image = nib.load(image_path)
    image_array = np.array(image.get_data())

    ## as MRI is float for original sigals, here we cast the float to uint8, range from 0 - 255
    image_array = (image_array - image_array.min())/(image_array.max() - image_array.min()) * 255
    image_array = image_array.astype('uint8')

    slice_to_rgb_imgs = []
    # slice_list = range(15, image_array.shape[view] - 15) # delete the first 20 slice and last 15 slices
    slice_list = range(70, 71) # for test

    if rgb_mode == 'three_slices':

        for i in slice_list:
            ## sagital
            if view == 0:
                slice_select_0 = image_array[i - 1, :, :]
                slice_select_1 = image_array[i, :, :]
                slice_select_2 = image_array[i + 1, :, :]

            ## coronal
            if view == 1:
                slice_select_0 = image_array[:, i - 1, :]
                slice_select_1 = image_array[:, i, :]
                slice_select_2 = image_array[:, i + 1, :]

            ## axial
            if view == 2:
                slice_select_0 = image_array[:, :, i - 1]
                slice_select_1 = image_array[:, :, i]
                slice_select_2 = image_array[:, :, i + 1]
    elif rgb_mode == 'single_slice':

        for i in slice_list:
            ## sagital
            if view == 0:
                slice_select_0 = image_array[i, :, :]
                slice_select_1 = image_array[i, :, :]
                slice_select_2 = image_array[i, :, :]

            ## coronal
            if view == 1:
                slice_select_0 = image_array[:, i, :]
                slice_select_1 = image_array[:, i, :]
                slice_select_2 = image_array[:, i, :]

            ## axial
            if view == 2:
                slice_select_0 = image_array[:, :, i]
                slice_select_1 = image_array[:, :, i]
                slice_select_2 = image_array[:, :, i]
    else:
        raise ValueError("Not yet implemented with this RGB methods")

    ## TODO, need to solve how to correctly convert slices into a RGB image without losting the nature of the image.
    # slice_to_rgb_img = np.stack((slice_select_0, slice_select_1, slice_select_2), axis=2)

    slice_to_rgb_img = np.zeros((slice_select_0.shape[0], slice_select_0.shape[1], 3))
    slice_to_rgb_img[..., 0] = slice_select_0
    slice_to_rgb_img[..., 1] = slice_select_1
    slice_to_rgb_img[..., 2] = slice_select_2

    if len(slice_to_rgb_img.shape) > 3 and slice_to_rgb_img.shape[3] == 1:
        slice_to_rgb_img_resize = np.resize(slice_to_rgb_img,
                                       (slice_to_rgb_img.shape[0], slice_to_rgb_img.shape[1], slice_to_rgb_img.shape[2]))
        slice_to_rgb_imgs.append(slice_to_rgb_img_resize)
    else:
        slice_to_rgb_imgs.append(slice_to_rgb_img)

    return slice_to_rgb_imgs

class CustomResize(object):
    def __init__(self, trg_size):
        self.trg_size = trg_size

    def __call__(self, img):
        resized_img = self.resize_image(img, self.trg_size)
        return resized_img

    def resize_image(self, img_array, trg_size):
        res = resize(img_array, trg_size, mode='reflect', preserve_range=True, anti_aliasing=False)

        # type check
        if type(res) != np.ndarray:
            raise "type error!"

        # PIL image cannot handle 3D image, only return ndarray type, which ToTensor accepts
        return res


class CustomToTensor(object):
    def __init__(self):
        pass

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            # backward compatibility
            return img.float()

def results_to_tsvs(output_dir, iteration, subject_list, y_truth, y_hat):
    """
    This is a function to trace all subject during training, test and validation, and calculate the performances with different metrics into tsv files.
    :param output_dir:
    :param iteration:
    :param subject_list:
    :param y_truth:
    :param y_hat:
    :return:
    """

    # check if the folder exist
    iteration_dir = os.path.join(output_dir, 'performances', 'iteration-' + str(iteration))
    if not os.path.exists(iteration_dir):
        os.makedirs(iteration_dir)
    iteration_subjects_df = pd.DataFrame({'iteration': iteration,
                                                'y': y_truth,
                                                'y_hat': y_hat,
                                                'subject': subject_list})
    iteration_subjects_df.to_csv(os.path.join(iteration_dir, 'subjects.tsv'), index=False, sep='\t', encoding='utf-8')

    results = evaluate_prediction(np.asarray(y_truth), np.asarray(y_hat))
    del results['confusion_matrix']
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(iteration_dir, 'result.tsv'), index=False, sep='\t', encoding='utf-8')

    return iteration_subjects_df, pd.DataFrame(results, index=[0])
