import torch
from torch.autograd import Variable
import nibabel as nib
from torch.utils.data import Dataset
import random
import os, shutil
from skimage.transform import resize
from os import path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import time

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

def train(model, data_loader, use_cuda, loss_func, optimizer, writer, epoch_i, model_mode="train", global_steps=0):
    """
    This is the function to train, validate or test the model, depending on the model_mode parameter.
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
    batch_time = AverageMeter()

    if model_mode == "train":
        model.train() ## set the model to training mode
    else:
        model.eval() ## set the model to evaluation mode
    end = time.time()
    for i, subject_data in enumerate(data_loader):
        # for each iteration, the train data contains batch_size * n_slices_in_each_subject images
        loss_batch = 0.0
        acc_batch = 0.0
        num_slice = len(subject_data)
        for j in range(num_slice):
            data_dic = subject_data[j]
            if use_cuda:
                imgs, labels = data_dic['image'].cuda(), data_dic['label'].cuda()
            else:
                imgs, labels = data_dic['image'], data_dic['label']

            ## add the participant_id + session_id
            image_ids = data_dic['image_id']
            subjects.extend(image_ids)

            # TO track of indices, int64 is a better choice for large models.
            integer_encoded = labels.data.cpu().numpy()
            gound_truth_list = integer_encoded.tolist()
            y_ground.extend(gound_truth_list)
            ground_truth = Variable(torch.from_numpy(integer_encoded)).long()

            print('The group true label is %s' % (str(labels)))
            if use_cuda:
                ground_truth = ground_truth.cuda()
            output = model(imgs)
            _, predict = output.topk(1)
            predict_list = predict.data.cpu().numpy().tolist()
            y_hat.extend([item for sublist in predict_list for item in sublist])
            if model_mode == "train" or model_mode == 'valid':
                print("output.device: " + str(output.device))
                print("ground_truth.device: " + str(ground_truth.device))
                loss = loss_func(output, ground_truth)
                loss_batch += loss
            correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
            correct_cnt += correct_this_batch
            # To monitor the training process using tensorboard, we only display the training loss and accuracy, the other performance metrics, such
            # as balanced accuracy, will be saved in the tsv file.
            accuracy = float(correct_this_batch) / len(ground_truth)
            acc_batch += accuracy
            if model_mode == "train":
                print("For batch %d slice %d training loss is : %f" % (i, j, loss.item()))
                print("For batch %d slice %d training accuracy is : %f" % (i, j, accuracy))
            elif model_mode == "valid":
                print("For batch %d slice %d validation accuracy is : %f" % (i, j, accuracy))
                print("For batch %d slice %d validation loss is : %f" % (i, j, loss.item()))
            elif model_mode == "test":
                print("For batch %d slice %d validate accuracy is : %f" % (i, j, accuracy))

            # Unlike tensorflow, in Pytorch, we need to manully zero the graident before each backpropagation step, becase Pytorch accumulates the gradients
            # on subsequent backward passes. The initial designing for this is convenient for training RNNs.
            if model_mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time))

        if model_mode == "train":
            writer.add_scalar('slice-level accuracy', acc_batch / num_slice, i + epoch_i * len(data_loader.dataset))
            writer.add_scalar('loss', loss_batch / num_slice, i + epoch_i * len(data_loader.dataset))
            ## There are bug for TensorboardX, which does not support 1 channel image
            writer.add_image('example_image', imgs, i + epoch_i * len(data_loader.dataset))
        elif model_mode == "test":
            writer.add_scalar('slice-level accuracy', acc_batch / num_slice, i)

        ## add all accuracy for each iteration
        acc += acc_batch / num_slice

    acc_mean = acc / len(data_loader)
    if model_mode == "valid":
        writer.add_scalar('slice-level accuracy', acc_mean, global_steps)

    if model_mode == "train":
        global_steps = i + epoch_i * len(data_loader.dataset)
    else:
        global_steps = 0

    return imgs, subjects, y_ground, y_hat, acc_mean, global_steps

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

class mri_to_slice_level(Dataset):
    """
    This class reads the CAPS of image processing pipeline of DL

    To note, this class processes the MRI to be RGB for transfer learning.

    Return: a Pytorch Dataset objective
    """

    def __init__(self, caps_directory, tsv, transform=None, transfer_learning=True):
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
        self.transfer_learning = transfer_learning

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

        if self.transfer_learning == True:
            ### get all the slices from the three view, basically, we take 3 slices and create a RBG image so that we can using this for transferring learning.
            ## For axial view
            axial_image_list = slices_to_rgb(image_path, 2, img_mode='rgb_slice')
            ## For coronal view
            coronal_image_list = slices_to_rgb(image_path, 1, img_mode='rgb_slice')
            ## For saggital view
            saggital_image_list = slices_to_rgb(image_path, 0, img_mode='rgb_slice')

            for img_rgb in (axial_image_list, coronal_image_list, saggital_image_list):
                for img in img_rgb:
                    if self.transform:
                        img = self.transform(img)
                    sample = {'image_id': img_name + '_' + sess_name, 'image': img, 'label': label}
                    samples.append(sample)

        else:
            ## For axial view
            axial_image_list = slices_to_rgb(image_path, 2, img_mode='original_slice')
            for img in (axial_image_list):
                if self.transform:
                    img = self.transform(img)
                sample = {'image_id': img_name + '_' + sess_name, 'image': img, 'label': label}
                samples.append(sample)

        random.shuffle(samples)

        return samples


def slices_to_rgb(image_path, view, img_mode='rgb_slice'):
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
    if img_mode == 'rgb_slice':
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
        # image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())

    slice_to_rgb_imgs = []
    # slice_list = range(15, image_array.shape[view] - 15) # delete the first 20 slice and last 15 slices
    slice_list = range(70, 71) # for test

    if img_mode == 'rgb_slice' or img_mode == "original_slice":

        for i in slice_list:
            ## sagital
            if view == 0:
                slice_select = image_array[i, :, :]

            ## coronal
            if view == 1:
                slice_select = image_array[:, i, :]

            ## axial
            if view == 2:
                slice_select = image_array[:, :, i]
    else:
        raise ValueError("Not yet implemented")

    if img_mode == 'original_slice':
        slice_to_rgb_img = np.reshape(slice_select, (slice_select.shape[0], slice_select.shape[1], 1))

        if len(slice_to_rgb_img.shape) > 3 and slice_to_rgb_img.shape[3] == 1:
            slice_to_rgb_img_resize = np.resize(slice_to_rgb_img,
                                           (slice_to_rgb_img.shape[0], slice_to_rgb_img.shape[1], slice_to_rgb_img.shape[2]))
            slice_to_rgb_imgs.append(slice_to_rgb_img_resize)
        else:
            slice_to_rgb_imgs.append(slice_to_rgb_img)

    else:
        # test = np.zeros((slice_select.shape[0], slice_select.shape[1], 3), dtype=np.float32)
        # test[..., 0] = slice_select
        # test[..., 1] = slice_select
        # test[..., 2] = slice_select

        slice_to_rgb_img = np.stack((slice_select,)*3, axis=-1)
        ## change the datatype into uint8, but before fitting the image into pytorch, pytorch needs float, that is why the contrast of image has been inversed.

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
        res = res.astype('uint8')
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
            img = torch.from_numpy(pic.transpose((2, 0, 1))).float()

            # Pytorch does not work with int type. Here, it just change the visualization, the value itself does not change.
            # return img.float()
            return img

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

def evaluate_prediction(y, y_hat):

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count