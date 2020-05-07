# coding: utf8

import torch
import pandas as pd
import numpy as np
from os import path
from torch.utils.data import Dataset, sampler
from scipy.ndimage.filters import gaussian_filter


#################################
# Datasets loaders
#################################

class MRIDataset(Dataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self, img_dir, data_file, preprocessing='linear', transform=None):
        """
        Args:
            img_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            preprocessing (string): Defines the path to the data in CAPS
            transform (callable, optional): Optional transform to be applied on a sample.

        """
        self.img_dir = img_dir
        self.transform = transform
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1, 'unlabeled': -1}
        self.data_path = preprocessing

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument datafile is not of correct type.')

        if ('diagnosis' not in list(self.df.columns.values)) or ('session_id' not in list(self.df.columns.values)) or \
           ('participant_id' not in list(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'diagnosis']")

        self.size = self[0]['image'].numpy().size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'participant_id']
        img_label = self.df.loc[idx, 'diagnosis']
        sess_name = self.df.loc[idx, 'session_id']
        # Not in BIDS but in CAPS
        if self.data_path == "linear":
            image_path = path.join(self.img_dir, 'subjects', img_name, sess_name,
                                   'deeplearning_prepare_data', 'image_based', 't1_linear',
                                   img_name + '_' + sess_name
                                   + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt')
        elif self.data_path == "mni":
            image_path = path.join(self.img_dir, 'subjects', img_name, sess_name,
                                   't1', 'spm', 'segmentation', 'normalized_space',
                                   img_name + '_' + sess_name + '_space-Ixi549Space_T1w.pt')
        else:
            raise NotImplementedError("The data path %s is not implemented" % self.data_path)

        image = torch.load(image_path)
        label = self.diagnosis_code[img_label]

        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label, 'participant_id': img_name, 'session_id': sess_name,
                  'image_path': image_path}

        return sample

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


class MRIDataset_patch(Dataset):

    def __init__(self, caps_directory, data_file, patch_size, stride_size, transformations=None, prepare_dl=False,
                 patch_index=None):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transformations (callable, optional): Optional transformations to be applied on a sample.

        """
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1}
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.prepare_dl = prepare_dl
        self.patch_index = patch_index

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument datafile is not of correct type.')

        if ('diagnosis' not in list(self.df.columns.values)) or ('session_id' not in list(self.df.columns.values)) or \
           ('participant_id' not in list(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'diagnosis']")

        self.patchs_per_patient = self.num_patches_per_session()

    def __len__(self):
        return len(self.df) * self.patchs_per_patient

    def __getitem__(self, idx):
        sub_idx = idx // self.patchs_per_patient
        img_name = self.df.loc[sub_idx, 'participant_id']
        sess_name = self.df.loc[sub_idx, 'session_id']
        img_label = self.df.loc[sub_idx, 'diagnosis']
        label = self.diagnosis_code[img_label]
        if self.patch_index is None:
            patch_idx = idx % self.patchs_per_patient
        else:
            patch_idx = self.patch_index

        if self.prepare_dl:
            patch_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                    'deeplearning_prepare_data', 'patch_based', 't1_linear',
                    img_name + '_' + sess_name
                    + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w'
                    + '_patchsize-' + str(self.patch_size)
                    + '_stride-' + str(self.stride_size)
                    + '_patch-' + str(patch_idx) + '.pt')

            patch = torch.load(patch_path)
        else:
            image_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                    'deeplearning_prepare_data', 'image_based', 't1_linear',
                    img_name + '_' + sess_name
                    + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt')
            image = torch.load(image_path)
            patch = extract_patch_from_mri(image, patch_idx, self.patch_size, self.stride_size)

        # check if the patch has NaN value
        if torch.isnan(patch).any():
            print("Double check, this patch has NaN value: %s" % str(img_name + '_' + sess_name + str(patch_idx)))
            patch[torch.isnan(patch)] = 0

        if self.transformations:
            patch = self.transformations(patch)

        sample = {'image_id': img_name + '_' + sess_name + '_patch' + str(patch_idx),
                'image': patch, 'label': label,
                'participant_id': img_name, 'session_id': sess_name,
                'patch_id': patch_idx}

        return sample

    def num_patches_per_session(self):
        if self.patch_index is not None:
            return 1

        img_name = self.df.loc[0, 'participant_id']
        sess_name = self.df.loc[0, 'session_id']

        image_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                'deeplearning_prepare_data', 'image_based', 't1_linear',
                img_name + '_' + sess_name
                + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt')
        image = torch.load(image_path)

        patches_tensor = image.unfold(1, self.patch_size, self.stride_size
                                      ).unfold(2, self.patch_size, self.stride_size
                                               ).unfold(3, self.patch_size, self.stride_size).contiguous()
        patches_tensor = patches_tensor.view(-1, self.patch_size, self.patch_size, self.patch_size)
        num_patches = patches_tensor.shape[0]
        return num_patches


class MRIDataset_patch_hippocampus(Dataset):

    def __init__(self, caps_directory, data_file, transformations=None):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transformations (callable, optional): Optional transformations to be applied on a sample.

        """
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.diagnosis_code = {'CN': 0, 'AD': 1, 'sMCI': 0, 'pMCI': 1, 'MCI': 1}

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument datafile is not of correct type.')

        if ('diagnosis' not in list(self.df.columns.values)) or ('session_id' not in list(self.df.columns.values)) or \
           ('participant_id' not in list(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'diagnosis']")

        self.patchs_per_patient = 2

    def __len__(self):
        return len(self.df) * self.patchs_per_patient

    def __getitem__(self, idx):
        sub_idx = idx // self.patchs_per_patient
        img_name = self.df.loc[sub_idx, 'participant_id']
        sess_name = self.df.loc[sub_idx, 'session_id']
        img_label = self.df.loc[sub_idx, 'diagnosis']
        label = self.diagnosis_code[img_label]

        # 1 is left hippocampus, 0 is right
        left_is_odd = idx % self.patchs_per_patient

        if left_is_odd == 1:
            patch_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                    't1', 'preprocessing_dl',
                    img_name + '_' + sess_name + '_space-MNI_res-1x1x1_hippocampus_hemi-left.pt')
        else:
            patch_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                    't1', 'preprocessing_dl',
                    img_name + '_' + sess_name + '_space-MNI_res-1x1x1_hippocampus_hemi-right.pt')

        patch = torch.load(patch_path)

        # check if the patch has NaN value
        if torch.isnan(patch).any():
            print("Double check, this patch has NaN value: %s" % str(img_name + '_' + sess_name + str(left_is_odd)))
            patch[torch.isnan(patch)] = 0

        if self.transformations:
            patch = self.transformations(patch)

        sample = {'image_id': img_name + '_' + sess_name + '_patch' + str(left_is_odd),
                'image': patch, 'label': label,
                'participant_id': img_name, 'session_id': sess_name,
                'patch_id': left_is_odd}

        return sample


class MRIDataset_slice(Dataset):
    """
    This class reads the CAPS of image processing pipeline of DL

    To note, this class processes the MRI to be RGB for transfer learning.

    Return: a Pytorch Dataset objective
    """

    def __init__(self, caps_directory, data_file, transformations=None, mri_plane=0, prepare_dl=False):
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
        self.prepare_dl = prepare_dl

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument datafile is not of correct type.')

        # This dimension is for the output of image processing pipeline of Raw: 169 * 208 * 179
        if mri_plane == 0:
            self.slices_per_patient = 169 - 40
            self.slice_direction = 'sag'
        elif mri_plane == 1:
            self.slices_per_patient = 208 - 40
            self.slice_direction = 'cor'
        elif mri_plane == 2:
            self.slices_per_patient = 179 - 40
            self.slice_direction = 'axi'

    def __len__(self):
        return len(self.df) * self.slices_per_patient

    def __getitem__(self, idx):
        sub_idx = idx // self.slices_per_patient
        img_name = self.df.loc[sub_idx, 'participant_id']
        sess_name = self.df.loc[sub_idx, 'session_id']
        img_label = self.df.loc[sub_idx, 'diagnosis']
        label = self.diagnosis_code[img_label]
        slice_idx = idx % self.slices_per_patient

        if self.prepare_dl:
            # read the slices directly
            slice_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                    'deeplearning_prepare_data', 'slice_based', 't1_linear',
                    img_name + '_' + sess_name
                    + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w' +
                    self.slice_direction + '_rgbslice-' + str(slice_idx + 20) + '.pt')
            extracted_slice = torch.load(slice_path)
        else:
            image_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                    'deeplearning_prepare_data', 'image_based', 't1_linear',
                    img_name + '_' + sess_name
                    + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt')
            image = torch.load(image_path)
            extracted_slice = extract_slice_from_mri(image, slice_idx + 20, self.mri_plane)

        # check if the slice has NaN value
        if torch.isnan(extracted_slice).any():
            print("Slice %s has NaN values." % str(img_name + '_' + sess_name + '_' + str(slice_idx + 20)))
            extracted_slice[torch.isnan(extracted_slice)] = 0

        if self.transformations:
            extracted_slice = self.transformations(extracted_slice)

        sample = {'image_id': img_name + '_' + sess_name + '_slice' + str(slice_idx + 20),
                'image': extracted_slice, 'label': label,
                'participant_id': img_name, 'session_id': sess_name,
                'slice_id': slice_idx + 20}

        return sample


class MRIDataset_slice_mixed(Dataset):
    """
    This class reads the CAPS of image processing pipeline of DL. However, this is used for the bad data split strategy

    To note, this class processes the MRI to be RGB for transfer learning.

    Return: a Pytorch Dataset objective
    """

    def __init__(self, caps_directory, data_file, transformations=None, mri_plane=0, prepare_dl=False):
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
        self.prepare_dl = prepare_dl

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument datafile is not of correct type.')

        if mri_plane == 0:
            self.slice_direction = 'sag'
        elif mri_plane == 1:
            self.slice_direction = 'cor'
        elif mri_plane == 2:
            self.slice_direction = 'axi'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'participant_id']
        sess_name = self.df.loc[idx, 'session_id']
        slice_name = self.df.loc[idx, 'slice_id']
        img_label = self.df.loc[idx, 'diagnosis']
        label = self.diagnosis_code[img_label]

        if self.prepare_dl:
            slice_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                    'deeplearning_prepare_data', 'slice_based', 't1_linear',
                    img_name + '_' + sess_name
                    + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w' +
                    self.slice_direction + '_rgbslice-' + str(slice_name) + '.pt')
            extracted_slice = torch.load(slice_path)

        else:
            image_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                    'deeplearning_prepare_data', 'image_based', 't1_linear',
                    img_name + '_' + sess_name
                    + '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt')
            extracted_slice = extract_slice_from_mri(image, slice_name, self.mri_plane)

        # check if the slice has NaN value
        if torch.isnan(extracted_slice).any():
            print("Slice %s has NaN values." % str(img_name + '_' + sess_name + '_' + str(slice_name)))
            extracted_slice[torch.isnan(extracted_slice)] = 0

        if self.transformations:
            extracted_slice = self.transformations(extracted_slice)

        sample = {'image_id': img_name + '_' + sess_name + '_slice' + str(slice_name),
                'image': extracted_slice, 'label': label,
                'participant_id': img_name, 'session_id': sess_name,
                'slice_id': slice_name}

        return sample


def extract_slice_from_mri(image, index_slice, view):
    """
    This is a function to grab one slice in each view and create a rgb image for transferring learning: duplicate the slices into R, G, B channel
    :param image: (tensor)
    :param index_slice: (int) index of the wanted slice
    :param view:
    :return:
    To note, for each view:
    Axial_view = "[:, :, slice_i]"
    Coronal_view = "[:, slice_i, :]"
    Sagittal_view= "[slice_i, :, :]"
    """

    # reshape the tensor, delete the first dimension for slice-level
    image_tensor = image.squeeze(0)

    # sagittal
    if view == 0:
        slice_select = image_tensor[index_slice, :, :].clone()

    # coronal
    elif view == 1:
        slice_select = image_tensor[:, index_slice, :].clone()

    # axial
    elif view == 2:
        slice_select = image_tensor[:, :, index_slice].clone()

    else:
        raise ValueError("This view does not exist, please choose view in [0, 1, 2]")

    extracted_slice = torch.stack((slice_select, slice_select, slice_select))

    return extracted_slice


def extract_patch_from_mri(image_tensor, index_patch, patch_size, stride_size):

    # use classifiers tensor.upfold to crop the patch.
    patches_tensor = image_tensor.unfold(1, patch_size, stride_size
                                         ).unfold(2, patch_size, stride_size
                                                  ).unfold(3, patch_size, stride_size).contiguous()
    patches_tensor = patches_tensor.view(-1, patch_size, patch_size, patch_size)
    extracted_patch = patches_tensor[index_patch, ...].unsqueeze_(0).clone()

    return extracted_patch


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


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


def load_data(train_val_path, diagnoses_list, split, n_splits=None, baseline=True):

    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()

    if n_splits is None:
        train_path = path.join(train_val_path, 'train')
        valid_path = path.join(train_val_path, 'validation')

    else:
        train_path = path.join(train_val_path, 'train_splits-' + str(n_splits),
                               'split-' + str(split))
        valid_path = path.join(train_val_path, 'validation_splits-' + str(n_splits),
                               'split-' + str(split))

    print("Train", train_path)
    print("Valid", valid_path)

    for diagnosis in diagnoses_list:

        if baseline:
            train_diagnosis_path = path.join(train_path, diagnosis + '_baseline.tsv')
        else:
            train_diagnosis_path = path.join(train_path, diagnosis + '.tsv')

        valid_diagnosis_path = path.join(valid_path, diagnosis + '_baseline.tsv')

        train_diagnosis_df = pd.read_csv(train_diagnosis_path, sep='\t')
        valid_diagnosis_df = pd.read_csv(valid_diagnosis_path, sep='\t')

        train_df = pd.concat([train_df, train_diagnosis_df])
        valid_df = pd.concat([valid_df, valid_diagnosis_df])

    train_df.reset_index(inplace=True, drop=True)
    valid_df.reset_index(inplace=True, drop=True)

    return train_df, valid_df


def load_data_test(test_path, diagnoses_list):

    test_df = pd.DataFrame()

    for diagnosis in diagnoses_list:

        test_diagnosis_path = path.join(test_path, diagnosis + '_baseline.tsv')
        test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep='\t')
        test_df = pd.concat([test_df, test_diagnosis_df])

    test_df.reset_index(inplace=True, drop=True)

    return test_df
