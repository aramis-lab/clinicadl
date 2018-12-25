import torch
import pandas as pd
import numpy as np
import os
from os import path
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from scipy.ndimage.filters import gaussian_filter
import random


class MRIDataset(Dataset):
    """labeled Faces in the Wild dataset."""

    def __init__(self, img_dir, data_file, patch_size, patch_stride, transform=None):
        """
        Args:
            img_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.

        """
        self.img_dir = img_dir
        self.transform = transform
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1}
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Check the format of the tsv file here
        self.df = pd.read_csv(data_file, sep='\t')
        if ('diagnosis' not in list(self.df.columns.values)) or ('session_id' not in list(self.df.columns.values)) or \
           ('participant_id' not in list(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'diagnosis']")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'participant_id']
        img_label = self.df.loc[idx, 'diagnosis']
        sess_name = self.df.loc[idx, 'session_id']
        # Not in BIDS but in CAPS
        image_path = path.join(self.img_dir, 'subjects', img_name, sess_name,
                               't1', 'preprocessing_dl',
                               img_name + '_' + sess_name + '_space-MNI_res-1x1x1.pt')
        image = torch.load(image_path)
        label = self.diagnosis_code[img_label]
        samples = []

        if self.transform:
            image = self.transform(image)

        ### extract the patch from MRI based on a specific size
        patches = extract_patches(image, self.patch_size, self.patch_stride)
        for patch in patches:
            sample = {'image_id': img_name + '_' + sess_name, 'image': patch, 'label': label}
            samples.append(sample)

        return samples

    def session_restriction(self, session):
        """
            Allows to generate a new MRIDataset using some specific sessions only (mostly used for evaluation of test)

            :param session: (str) the session wanted. Must be 'all' or 'ses-MXX'
            :return: (DataFrame) the dataset with the wanted sessions
            """
        from copy import copy

        data_output = copy(self)
        if session == "all":
            return data_output
        else:
            df_session = self.df[self.df.session_id == session]
            df_session.reset_index(drop=True, inplace=True)
            data_output.df = df_session
            if len(data_output) == 0:
                raise Exception("The session %s doesn't exist for any of the subjects in the test data" % session)
            return data_output


class GaussianSmoothing(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        image = sample['image']
        np.nan_to_num(image, copy=False)
        smoothed_image = gaussian_filter(image, sigma=self.sigma)
        sample['image'] = smoothed_image

        return sample


class ToTensor(object):
    """Convert image type to Tensor and diagnosis to diagnosis code"""

    def __call__(self, image):
        np.nan_to_num(image, copy=False)
        image = image.astype(float)

        return torch.from_numpy(image[np.newaxis, :]).float()


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

def split_subjects_to_tsv(diagnoses_tsv, val_size=0.15):
    """
    Write the tsv files corresponding to the train/val/test splits of all folds

    :param diagnoses_tsv: (str) path to the tsv file with diagnoses
    :param val_size: (float) proportion of the train set being used for validation
    :return: None
    """

    df = pd.read_csv(diagnoses_tsv, sep='\t')
    if 'diagnosis' not in list(df.columns.values):
        raise Exception('Diagnoses file is not in the correct format.')
    # Here we reduce the DataFrame to have only one diagnosis per subject (multiple time points case)
    diagnosis_df = subject_diagnosis_df(df)
    diagnoses_list = list(diagnosis_df.diagnosis)
    unique = list(set(diagnoses_list))
    y = np.array([unique.index(x) for x in diagnoses_list])  # There is one label per diagnosis depending on the order

    sets_dir = path.join(path.dirname(diagnoses_tsv),
                         path.basename(diagnoses_tsv).split('.')[0],
                         'val_size-' + str(val_size))
    if not path.exists(sets_dir):
        os.makedirs(sets_dir)

    # split the train data into training and validation set
    skf_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
    indices = next(skf_2.split(np.zeros(len(y)), y))
    train_ind, valid_ind = indices

    df_sub_valid = diagnosis_df.iloc[valid_ind]
    df_sub_train = diagnosis_df.iloc[train_ind]
    df_valid = multiple_time_points(df, df_sub_valid)
    df_train = multiple_time_points(df, df_sub_train)

    df_valid.to_csv(path.join(sets_dir, 'valid.tsv'), sep='\t', index=False)
    df_train.to_csv(path.join(sets_dir, 'train.tsv'), sep='\t', index=False)


# def old_load_split(diagnoses_tsv, fold, n_splits=5, val_size=0.15):
#     """
#     Returns the paths of the TSV files for each set
#
#     :param diagnoses_tsv: (str) path to the tsv file with diagnoses
#     :param fold: (int) the number of the current fold
#     :param n_splits: (int) the total number of folds
#     :param val_size: (float) the proportion of the training set used for validation
#     :return: 3 Strings
#         training_tsv
#         test_tsv
#         valid_tsv
#     """
#     sets_dir = path.join(path.dirname(diagnoses_tsv),
#                          path.basename(diagnoses_tsv).split('.')[0],
#                          'splits-' + str(n_splits))
#
#     if fold >= n_splits:
#         raise Exception("The fold number must not exceed the number of splits.")
#
#     training_tsv = path.join(sets_dir,
#                              'val_size-' + str(val_size) + '_iteration-' + str(fold) + '_train.tsv')
#     test_tsv = path.join(sets_dir,
#                          'val_size-' + str(val_size) + '_iteration-' + str(fold) + '_test.tsv')
#     valid_tsv = path.join(sets_dir,
#                           'val_size-' + str(val_size) + '_iteration-' + str(fold) + '_valid.tsv')
#
#     if not path.exists(training_tsv) or not path.exists(test_tsv) or not path.exists(valid_tsv):
#         old_split_subjects_to_tsv(diagnoses_tsv, n_splits, val_size)
#
#         training_tsv = path.join(sets_dir,
#                                  'val_size-' + str(val_size) + '_iteration-' + str(fold) + '_train.tsv')
#         test_tsv = path.join(sets_dir,
#                              'val_size-' + str(val_size) + '_iteration-' + str(fold) + '_test.tsv')
#         valid_tsv = path.join(sets_dir,
#                               'val_size-' + str(val_size) + '_iteration-' + str(fold) + '_valid.tsv')
#
#     return training_tsv, test_tsv, valid_tsv


def load_split(diagnoses_tsv, val_size=0.15):
    """
    Returns the paths of the TSV files for each set

    :param diagnoses_tsv: (str) path to the tsv file with diagnoses
    :param val_size: (float) the proportion of the training set used for validation
    :return: 3 Strings
        training_tsv
        valid_tsv
    """
    sets_dir = path.join(path.dirname(diagnoses_tsv),
                         path.basename(diagnoses_tsv).split('.')[0],
                         'val_size-' + str(val_size))

    training_tsv = path.join(sets_dir, 'train.tsv')
    valid_tsv = path.join(sets_dir, 'valid.tsv')

    if not path.exists(training_tsv) or not path.exists(valid_tsv):
        split_subjects_to_tsv(diagnoses_tsv, val_size)

        training_tsv = path.join(sets_dir, 'train.tsv')
        valid_tsv = path.join(sets_dir, 'valid.tsv')

    return training_tsv, valid_tsv


def extract_patches(image_tensor, patch_size, stride):

    ## use pytorch tensor.upfold to crop the patch.
    patches_tensor = image_tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride).unfold(3, patch_size, stride).contiguous()
    # the dimension of patch_tensor should be [1, patch_num1, patch_num2, patch_num3, patch_size1, patch_size2, patch_size3]
    # resize it into [num_patch, patch_size1, patch_size2, patch_size3]
    patches_tensor = patches_tensor.view(-1, patch_size, patch_size, patch_size)
    patches = []
    for i in range(patches_tensor.shape[0]):
        patch = patches_tensor[i, ...]
        patch.unsqueeze_(0)
        patches.append(patch)
    return patches

class customToTensor(object):
    def __init__(self):
        pass

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1))).float()

            # Pytorch does not work with int type. Here, it just change the visualization, the value itself does not change.
            # return img.float()
            return img