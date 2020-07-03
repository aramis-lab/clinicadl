# coding: utf8

import torch
import pandas as pd
import numpy as np
from os import path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import abc
from clinicadl.tools.inputs.filename_types import FILENAME_TYPE


#################################
# Datasets loaders
#################################


class MRIDataset(Dataset):
    """Abstract class for all derived MRIDatasets."""
    def __init__(self, caps_directory, data_file, preprocessing, transformations=None):
        self.caps_directory = caps_directory
        self.transformations = transformations
        self.diagnosis_code = {
            'CN': 0,
            'AD': 1,
            'sMCI': 0,
            'pMCI': 1,
            'MCI': 1,
            'unlabeled': -1}
        self.preprocessing = preprocessing

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument data_file is not of correct type.')

        if ('diagnosis' not in list(self.df.columns.values)) or ('session_id' not in list(self.df.columns.values)) or \
           ('participant_id' not in list(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'diagnosis']")

        self.elem_per_image = self.num_elem_per_image()

    def __len__(self):
        return len(self.df) * self.elem_per_image

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    @abc.abstractmethod
    def num_elem_per_image(self):
        pass


class MRIDatasetImage(MRIDataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self, caps_directory, data_file,
                 preprocessing='t1-linear', transformations=None):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            preprocessing (string): Defines the path to the data in CAPS
            transform (callable, optional): Optional transform to be applied on a sample.

        """
        super().__init__(caps_directory, data_file, preprocessing, transformations)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'participant_id']
        img_label = self.df.loc[idx, 'diagnosis']
        sess_name = self.df.loc[idx, 'session_id']
        # Not in BIDS but in CAPS
        if self.preprocessing == "t1-linear":
            image_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                                   'deeplearning_prepare_data', 'image_based', 't1_linear',
                                   img_name + '_' + sess_name
                                   + FILENAME_TYPE['cropped'] + '.pt')
        elif self.preprocessing == "t1-extensive":
            image_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                                   't1', 'spm', 'segmentation', 'normalized_space',
                                   img_name + '_' + sess_name
                                   + FILENAME_TYPE['skull_stripped'] + '.pt')
        else:
            raise NotImplementedError(
                "The path to preprocessing %s is not implemented" % self.preprocessing)

        image = torch.load(image_path)
        label = self.diagnosis_code[img_label]

        if self.transformations:
            image = self.transformations(image)
        sample = {'image': image, 'label': label, 'participant_id': img_name, 'session_id': sess_name,
                  'image_path': image_path}

        return sample

    def num_elem_per_image(self):
        return 1


class MRIDatasetPatch(MRIDataset):

    def __init__(self, caps_directory, data_file, patch_size, stride_size, transformations=None, prepare_dl=False,
                 patch_index=None, preprocessing="t1-linear"):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transformations (callable, optional): Optional transformations to be applied on a sample.

        """
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.patch_index = patch_index
        super().__init__(caps_directory, data_file, preprocessing, transformations)
        self.prepare_dl = prepare_dl

        if self.preprocessing != "t1-linear":
            raise NotImplementedError("The preprocessing %s was not implemented for patches. "
                                      "Raise an issue on GitHub to propose it !" % self.preprocessing)

    def __getitem__(self, idx):
        sub_idx = idx // self.elem_per_image
        img_name = self.df.loc[sub_idx, 'participant_id']
        sess_name = self.df.loc[sub_idx, 'session_id']
        img_label = self.df.loc[sub_idx, 'diagnosis']
        label = self.diagnosis_code[img_label]
        if self.patch_index is None:
            patch_idx = idx % self.elem_per_image
        else:
            patch_idx = self.patch_index

        if self.prepare_dl:
            patch_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                                   'deeplearning_prepare_data', 'patch_based', 't1_linear',
                                   img_name + '_' + sess_name
                                   + FILENAME_TYPE['cropped'][0:-4]
                                   + '_patchsize-' + str(self.patch_size)
                                   + '_stride-' + str(self.stride_size)
                                   + '_patch-' + str(patch_idx) + '_T1w.pt')

            patch = torch.load(patch_path)
        else:
            image_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                                   'deeplearning_prepare_data', 'image_based', 't1_linear',
                                   img_name + '_' + sess_name
                                   + FILENAME_TYPE['cropped'] + '.pt')
            image = torch.load(image_path)
            patch = extract_patch_from_mri(
                image, patch_idx, self.patch_size, self.stride_size)

        # check if the patch has NaN value
        if torch.isnan(patch).any():
            print("Double check, this patch has NaN value: %s" %
                  str(img_name + '_' + sess_name + str(patch_idx)))
            patch[torch.isnan(patch)] = 0

        if self.transformations:
            patch = self.transformations(patch)

        sample = {'image_id': img_name + '_' + sess_name + '_patch' + str(patch_idx),
                  'image': patch, 'label': label,
                  'participant_id': img_name, 'session_id': sess_name,
                  'patch_id': patch_idx}

        return sample

    def num_elem_per_image(self):
        if self.patch_index is not None:
            return 1

        img_name = self.df.loc[0, 'participant_id']
        sess_name = self.df.loc[0, 'session_id']

        image_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                               'deeplearning_prepare_data', 'image_based', 't1_linear',
                               img_name + '_' + sess_name
                               + FILENAME_TYPE['cropped'] + '.pt')
        image = torch.load(image_path)

        patches_tensor = image.unfold(1, self.patch_size, self.stride_size
                                      ).unfold(2, self.patch_size, self.stride_size
                                               ).unfold(3, self.patch_size, self.stride_size).contiguous()
        patches_tensor = patches_tensor.view(-1,
                                             self.patch_size,
                                             self.patch_size,
                                             self.patch_size)
        num_patches = patches_tensor.shape[0]
        return num_patches


class MRIDatasetRoi(MRIDataset):

    def __init__(self, caps_directory, data_file, preprocessing="t1-linear",
                 transformations=None, prepare_dl=False):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transformations (callable, optional): Optional transformations to be applied on a sample.
            prepare_dl (bool): If True the outputs of extract preprocessing are used, else the whole
            MRI is loaded.

        """
        super().__init__(caps_directory, data_file, preprocessing, transformations)
        self.prepare_dl = prepare_dl

        if self.preprocessing != "t1-linear":
            raise NotImplementedError("The preprocessing %s was not implemented for ROI. "
                                      "Raise an issue on GitHub to propose it !" % self.preprocessing)

    def __getitem__(self, idx):
        sub_idx = idx // self.elem_per_image
        img_name = self.df.loc[sub_idx, 'participant_id']
        sess_name = self.df.loc[sub_idx, 'session_id']
        img_label = self.df.loc[sub_idx, 'diagnosis']
        label = self.diagnosis_code[img_label]

        # 1 is left hippocampus, 0 is right
        left_is_odd = idx % self.elem_per_image
        if self.prepare_dl:
            raise NotImplementedError(
                'The extraction of ROIs prior to training is not implemented.')

        else:
            image_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                                   'deeplearning_prepare_data', 'image_based', 't1_linear',
                                   img_name + '_' + sess_name
                                   + FILENAME_TYPE['cropped'] + '.pt')
            image = torch.load(image_path)
            patch = extract_roi_from_mri(image, left_is_odd)

        # check if the patch has NaN value
        if torch.isnan(patch).any():
            print("Double check, this patch has NaN value: %s" %
                  str(img_name + '_' + sess_name + str(left_is_odd)))
            patch[torch.isnan(patch)] = 0

        if self.transformations:
            patch = self.transformations(patch)

        sample = {'image_id': img_name + '_' + sess_name + '_patch' + str(left_is_odd),
                  'image': patch, 'label': label,
                  'participant_id': img_name, 'session_id': sess_name,
                  'roi_id': left_is_odd}

        return sample

    def num_elem_per_image(self):
        return 2


class MRIDatasetSlice(MRIDataset):
    """
    This class reads the CAPS of image processing pipeline of DL

    To note, this class processes the MRI to be RGB for transfer learning.

    Return: a Pytorch Dataset objective
    """

    def __init__(self, caps_directory, data_file, preprocessing="t1-linear",
                 transformations=None, mri_plane=0, prepare_dl=False,
                 discarded_slices=20):
        """
        Args:
            caps_directory (string): the output folder of image processing pipeline.
            transformations (callable, optional): if the data sample should be done some transformations or not, such as resize the image.
            discarded_slices (int or list): slices discarded at the beginning and the end of the volume.

        To note, for each view:
            Axial_view = "[:, :, slice_i]"
            Coronal_veiw = "[:, slice_i, :]"
            Saggital_view= "[slice_i, :, :]"

        """
        # Rename MRI plane
        self.mri_plane = mri_plane
        if mri_plane == 0:
            self.slice_direction = 'sag'
        elif mri_plane == 1:
            self.slice_direction = 'cor'
        elif mri_plane == 2:
            self.slice_direction = 'axi'

        # Manage discarded_slices
        if type(discarded_slices) is int:
            discarded_slices = [discarded_slices, discarded_slices]
        if type(discarded_slices) is list and len(discarded_slices) == 1:
            discarded_slices = discarded_slices * 2
        self.discarded_slices = discarded_slices

        super().__init__(caps_directory, data_file, preprocessing, transformations)
        self.prepare_dl = prepare_dl

        if self.preprocessing != "t1-linear":
            raise NotImplementedError("The preprocessing %s was not implemented for slices. "
                                      "Raise an issue on GitHub to propose it !" % self.preprocessing)

    def __getitem__(self, idx):
        sub_idx = idx // self.elem_per_image
        img_name = self.df.loc[sub_idx, 'participant_id']
        sess_name = self.df.loc[sub_idx, 'session_id']
        img_label = self.df.loc[sub_idx, 'diagnosis']
        label = self.diagnosis_code[img_label]
        slice_idx = idx % self.elem_per_image + self.discarded_slices[0]

        if self.prepare_dl:
            # read the slices directly
            slice_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                                   'deeplearning_prepare_data', 'slice_based', 't1_linear',
                                   img_name + '_' + sess_name
                                   + FILENAME_TYPE['cropped'][0:-4]
                                   + '_axis-' + self.slice_direction
                                   + '_channel-rgb_slice-%i_T1w.pt' % slice_idx)
            extracted_slice = torch.load(slice_path)
        else:
            image_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                                   'deeplearning_prepare_data', 'image_based', 't1_linear',
                                   img_name + '_' + sess_name
                                   + FILENAME_TYPE['cropped'] + '.pt')
            image = torch.load(image_path)
            extracted_slice = extract_slice_from_mri(
                image, slice_idx, self.mri_plane)

        # check if the slice has NaN value
        if torch.isnan(extracted_slice).any():
            print("Slice %s has NaN values." %
                  str(img_name + '_' + sess_name + '_%i' % slice_idx))
            extracted_slice[torch.isnan(extracted_slice)] = 0

        if self.transformations:
            extracted_slice = self.transformations(extracted_slice)

        sample = {'image_id': img_name + '_' + sess_name + '_slice' + str(slice_idx),
                  'image': extracted_slice, 'label': label,
                  'participant_id': img_name, 'session_id': sess_name,
                  'slice_id': slice_idx + 20}

        return sample

    def num_elem_per_image(self):
        if self.slice_direction == 'sag':
            return 169 - self.discarded_slices[0] - self.discarded_slices[1]
        elif self.slice_direction == 'cor':
            return 208 - self.discarded_slices[0] - self.discarded_slices[1]
        elif self.slice_direction == 'axi':
            return 179 - self.discarded_slices[0] - self.discarded_slices[1]


class MRIDatasetSliceMixed(MRIDataset):
    """
    This class reads the CAPS of image processing pipeline of DL. However, this is used for the bad data split strategy

    To note, this class processes the MRI to be RGB for transfer learning.

    Return: a Pytorch Dataset objective
    """

    def __init__(self, caps_directory, data_file,
                 transformations=None, mri_plane=0, prepare_dl=False):
        """
        Args:
            caps_directory (string): the output folder of image processing pipeline.
            transformations (callable, optional): if the data sample should be done some transformations or not, such as resize the image.

        To note, for each view:
            Axial_view = "[:, :, slice_i]"
            Coronal_veiw = "[:, slice_i, :]"
            Saggital_view= "[slice_i, :, :]"

        """
        super().__init__(caps_directory, data_file, 't1-linear', transformations)
        self.mri_plane = mri_plane
        self.prepare_dl = prepare_dl

        if mri_plane == 0:
            self.slice_direction = 'sag'
        elif mri_plane == 1:
            self.slice_direction = 'cor'
        elif mri_plane == 2:
            self.slice_direction = 'axi'

        if self.preprocessing != "t1-linear":
            raise NotImplementedError("The preprocessing %s was not implemented for mixed slices. "
                                      "Raise an issue on GitHub to propose it !" % self.preprocessing)

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
                                   + FILENAME_TYPE['cropped'][0:-4]
                                   + '_axis-' + self.slice_direction
                                   + '_channel-rgb_slice-' + str(slice_name) + '_T1w.pt')
            extracted_slice = torch.load(slice_path)

        else:
            image_path = path.join(self.caps_directory, 'subjects', img_name, sess_name,
                                   'deeplearning_prepare_data', 'image_based', 't1_linear',
                                   img_name + '_' + sess_name
                                   + FILENAME_TYPE['cropped'] + '.pt')
            image = torch.load(image_path)
            extracted_slice = extract_slice_from_mri(
                image, slice_name, self.mri_plane)

        # check if the slice has NaN value
        if torch.isnan(extracted_slice).any():
            print("Slice %s has NaN values." %
                  str(img_name + '_' + sess_name + '_' + str(slice_name)))
            extracted_slice[torch.isnan(extracted_slice)] = 0

        if self.transformations:
            extracted_slice = self.transformations(extracted_slice)

        sample = {'image_id': img_name + '_' + sess_name + '_slice' + str(slice_name),
                  'image': extracted_slice, 'label': label,
                  'participant_id': img_name, 'session_id': sess_name,
                  'slice_id': slice_name}

        return sample

    def num_elem_per_image(self):
        return 1


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
        raise ValueError(
            "This view does not exist, please choose view in [0, 1, 2]")

    extracted_slice = torch.stack((slice_select, slice_select, slice_select))

    return extracted_slice


def extract_patch_from_mri(image_tensor, index_patch, patch_size, stride_size):

    # use classifiers tensor.upfold to crop the patch.
    patches_tensor = image_tensor.unfold(1, patch_size, stride_size
                                         ).unfold(2, patch_size, stride_size
                                                  ).unfold(3, patch_size, stride_size).contiguous()
    patches_tensor = patches_tensor.view(-1,
                                         patch_size,
                                         patch_size,
                                         patch_size)
    extracted_patch = patches_tensor[index_patch, ...].unsqueeze_(0).clone()

    return extracted_patch


def extract_roi_from_mri(image_tensor, left_is_odd):
    """

    :param image_tensor: (Tensor) the tensor of the image.
    :param left_is_odd: (int) if 1 the left hippocampus is extracted, else the right one.
    :return: Tensor of the extracted hippocampus
    """

    if left_is_odd == 1:
        crop_center = (61, 96, 68)  # the center of the left hippocampus
    else:
        crop_center = (109, 96, 68)  # the center of the right hippocampus
    crop_size = (50, 50, 50)  # the output cropped hippocampus size

    extracted_roi = image_tensor[
        :,
        crop_center[0] - crop_size[0] // 2: crop_center[0] + crop_size[0] // 2:,
        crop_center[1] - crop_size[1] // 2: crop_center[1] + crop_size[1] // 2:,
        crop_center[2] - crop_size[2] // 2: crop_center[2] + crop_size[2] // 2:
    ].clone()

    return extracted_roi


def return_dataset(mode, input_dir, data_df, preprocessing, transformations, params, cnn_index=None):
    """
    Return appropriate Dataset according to given options.

    Args:
        mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
        input_dir: (str) path to a directory containing a CAPS structure.
        data_df: (DataFrame) List subjects, sessions and diagnoses.
        preprocessing: (str) type of preprocessing wanted ('t1-linear' or 't1-extensive')
        transformations: (transforms) list of transformations performed on-the-fly.
        params: (Namespace) options used by specific modes.
        cnn_index: (int) Index of the CNN in a multi-CNN paradigm (optional).

    Returns:
         (Dataset) the corresponding dataset.
    """

    if cnn_index is not None and mode in ["image", "roi", "slice"]:
        raise ValueError("Multi-CNN is not implemented for %s mode." % mode)

    if mode == "image":
        return MRIDatasetImage(
            input_dir,
            data_df,
            preprocessing,
            transformations=transformations
        )
    if mode == "patch":
        return MRIDatasetPatch(
            input_dir,
            data_df,
            params.patch_size,
            params.stride_size,
            preprocessing=preprocessing,
            transformations=transformations,
            prepare_dl=params.prepare_dl,
            patch_index=cnn_index
        )
    elif mode == "roi":
        return MRIDatasetRoi(
            input_dir,
            data_df,
            preprocessing=preprocessing,
            transformations=transformations
        )
    elif mode == "slice":
        return MRIDatasetSlice(
            input_dir,
            data_df,
            preprocessing=preprocessing,
            transformations=transformations,
            mri_plane=params.mri_plane,
            prepare_dl=params.prepare_dl,
            discarded_slices=params.discarded_slices)
    else:
        raise ValueError("Mode %s is not implemented." % mode)


def compute_num_cnn(input_dir, tsv_path, options, data="train"):

    transformations = get_transforms(options.mode, options.minmaxnormalization)

    if data == "train":
        example_df, _ = load_data(tsv_path, options.diagnoses, 0, options.n_splits, options.baseline)
    elif data == "classify":
        example_df = pd.read_csv(tsv_path, sep='\t')
    else:
        example_df = load_data_test(tsv_path, options.diagnoses)

    full_dataset = return_dataset(options.mode, input_dir, example_df,
                                  options.preprocessing, transformations, options)

    return full_dataset.elem_per_image


##################################
# Transformations
##################################

class GaussianSmoothing(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        from scipy.ndimage.filters import gaussian_filter

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


def get_transforms(mode, minmaxnormalization=True):
    if mode in ["image", "patch", "roi"]:
        if minmaxnormalization:
            transformations = MinMaxNormalization()
        else:
            transformations = None
    elif mode == "slice":
        trg_size = (224, 224)
        if minmaxnormalization:
            transformations = transforms.Compose([MinMaxNormalization(),
                                                  transforms.ToPILImage(),
                                                  transforms.Resize(trg_size),
                                                  transforms.ToTensor()])
        else:
            transformations = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(trg_size),
                                                  transforms.ToTensor()])
    else:
        raise ValueError("Transforms for mode %s are not implemented." % mode)

    return transformations


################################
# tsv files loaders
################################

def load_data(train_val_path, diagnoses_list,
              split, n_splits=None, baseline=True):

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
            train_diagnosis_path = path.join(
                train_path, diagnosis + '_baseline.tsv')
        else:
            train_diagnosis_path = path.join(train_path, diagnosis + '.tsv')

        valid_diagnosis_path = path.join(
            valid_path, diagnosis + '_baseline.tsv')

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


def mix_slices(df_training, df_validation, mri_plane=0, val_size=0.15):
    """
    This is a function to gather the training and validation tsv together, then do the bad data split by slice.
    :param training_tsv:
    :param validation_tsv:
    :return:
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    df_all = pd.concat([df_training, df_validation])
    df_all = df_all.reset_index(drop=True)

    if mri_plane == 0:
        slices_per_patient = 169 - 40
        slice_index = list(np.arange(20, 169 - 20))
    elif mri_plane == 1:
        slices_per_patient = 208 - 40
        slice_index = list(np.arange(20, 208 - 20))
    else:
        slices_per_patient = 179 - 40
        slice_index = list(np.arange(20, 179 - 20))

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

    df_sub_train.reset_index(inplace=True, drop=True)
    df_sub_valid.reset_index(inplace=True, drop=True)

    return df_sub_train, df_sub_valid
