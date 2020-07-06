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

    def __init__(self, caps_directory, data_file,
                 preprocessing, transformations=None):
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

        if not hasattr(self, 'elem_index'):
            raise ValueError(
                "Child class of MRIDataset must set elem_index attribute.")
        if not hasattr(self, 'mode'):
            raise ValueError(
                "Child class of MRIDataset must set mode attribute.")

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument data_file is not of correct type.')

        mandatory_col = {"participant_id", "session_id", "diagnosis"}
        if self.elem_index == "mixed":
            mandatory_col.add("%s_id" % self.mode)

        if not mandatory_col.issubset(set(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include %s" % mandatory_col)

        self.elem_per_image = self.num_elem_per_image()

    def __len__(self):
        return len(self.df) * self.elem_per_image

    def _get_path(self, participant, session, mode="image"):

        if self.preprocessing == "t1-linear":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_linear',
                                   participant + '_' + session
                                   + FILENAME_TYPE['cropped'] + '.pt')
        elif self.preprocessing == "t1-extensive":
            image_path = path.join(self.caps_directory, 'subjects', participant, session,
                                   'deeplearning_prepare_data', '%s_based' % mode, 't1_extensive',
                                   participant + '_' + session
                                   + FILENAME_TYPE['skull_stripped'] + '.pt')
        else:
            raise NotImplementedError(
                "The path to preprocessing %s is not implemented" % self.preprocessing)

        return image_path

    def _get_meta_data(self, idx):
        image_idx = idx // self.elem_per_image
        participant = self.df.loc[image_idx, 'participant_id']
        session = self.df.loc[image_idx, 'session_id']

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        elif self.elem_index == "mixed":
            elem_idx = self.df.loc[image_idx, '%s_id' % self.mode]
        else:
            elem_idx = self.elem_index

        diagnosis = self.df.loc[image_idx, 'diagnosis']
        label = self.diagnosis_code[diagnosis]

        return participant, session, elem_idx, label

    def _get_full_image(self):
        from ..data.utils import find_image_path as get_nii_path
        import nibabel as nib

        participant_id = self.df.loc[0, 'participant_id']
        session_id = self.df.loc[0, 'session_id']

        try:
            image_path = self._get_path(participant_id, session_id, "image")
            image = torch.load(image_path)
        except FileNotFoundError:
            image_path = get_nii_path(
                self.caps_directory,
                participant_id,
                session_id,
                preprocessing=self.preprocessing)
            image_nii = nib.load(image_path)
            image_np = image_nii.get_fdata()
            image = ToTensor()(image_np)

        return image

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
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            transformations (callable, optional): Optional transform to be applied on a sample.

        """
        self.elem_index = None
        self.mode = "image"
        super().__init__(caps_directory, data_file, preprocessing, transformations)

    def __getitem__(self, idx):
        participant, session, _, label = self._get_meta_data(idx)

        image_path = self._get_path(participant, session, "image")
        image = torch.load(image_path)

        if self.transformations:
            image = self.transformations(image)
        sample = {'image': image, 'label': label, 'participant_id': participant, 'session_id': session,
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
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            transformations (callable, optional): Optional transform to be applied on a sample.
            prepare_dl (bool): If true pre-extracted patches will be loaded.
            patch_index (int, optional): If a value is given the same patch location will be extracted for each image.
                else the dataset will load all the patches possible for one image.
            patch_size (int): size of the regular cubic patch.
            stride_size (int): length between the centers of two patches.

        """
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.elem_index = patch_index
        self.mode = "patch"
        super().__init__(caps_directory, data_file, preprocessing, transformations)
        self.prepare_dl = prepare_dl

    def __getitem__(self, idx):
        participant, session, patch_idx, label = self._get_meta_data(idx)

        if self.prepare_dl:
            patch_path = path.join(self._get_path(participant, session, "patch")[0:-7]
                                   + '_patchsize-' + str(self.patch_size)
                                   + '_stride-' + str(self.stride_size)
                                   + '_patch-' + str(patch_idx) + '_T1w.pt')

            image = torch.load(patch_path)
        else:
            image_path = self._get_path(participant, session, "image")
            full_image = torch.load(image_path)
            image = self.extract_patch_from_mri(full_image, patch_idx)

        if self.transformations:
            image = self.transformations(image)

        sample = {'image': image, 'label': label,
                  'participant_id': participant, 'session_id': session, 'patch_id': patch_idx}

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1

        image = self._get_full_image()

        patches_tensor = image.unfold(1, self.patch_size, self.stride_size
                                      ).unfold(2, self.patch_size, self.stride_size
                                               ).unfold(3, self.patch_size, self.stride_size).contiguous()
        patches_tensor = patches_tensor.view(-1,
                                             self.patch_size,
                                             self.patch_size,
                                             self.patch_size)
        num_patches = patches_tensor.shape[0]
        return num_patches

    def extract_patch_from_mri(self, image_tensor, index_patch):

        patches_tensor = image_tensor.unfold(1, self.patch_size, self.stride_size
                                             ).unfold(2, self.patch_size, self.stride_size
                                                      ).unfold(3, self.patch_size, self.stride_size).contiguous()
        patches_tensor = patches_tensor.view(-1,
                                             self.patch_size,
                                             self.patch_size,
                                             self.patch_size)
        extracted_patch = patches_tensor[index_patch, ...].unsqueeze_(
            0).clone()

        return extracted_patch


class MRIDatasetRoi(MRIDataset):

    def __init__(self, caps_directory, data_file, preprocessing="t1-linear",
                 transformations=None, prepare_dl=False):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            transformations (callable, optional): Optional transform to be applied on a sample.
            prepare_dl (bool): If true pre-extracted patches will be loaded.

        """
        self.elem_index = None
        self.mode = "roi"
        super().__init__(caps_directory, data_file, preprocessing, transformations)
        self.prepare_dl = prepare_dl

    def __getitem__(self, idx):
        participant, session, roi_idx, label = self._get_meta_data(idx)

        if self.prepare_dl:
            raise NotImplementedError(
                'The extraction of ROIs prior to training is not implemented.')

        else:
            image_path = self._get_path(participant, session, "image")
            image = torch.load(image_path)
            patch = self.extract_roi_from_mri(image, roi_idx)

        if self.transformations:
            patch = self.transformations(patch)

        sample = {'image': patch, 'label': label,
                  'participant_id': participant, 'session_id': session,
                  'roi_id': roi_idx}

        return sample

    def num_elem_per_image(self):
        return 2

    def extract_roi_from_mri(self, image_tensor, left_is_odd):
        """

        :param image_tensor: (Tensor) the tensor of the image.
        :param left_is_odd: (int) if 1 the left hippocampus is extracted, else the right one.
        :return: Tensor of the extracted hippocampus
        """

        if self.preprocessing == "t1-linear":
            if left_is_odd == 1:
                # the center of the left hippocampus
                crop_center = (61, 96, 68)
            else:
                # the center of the right hippocampus
                crop_center = (109, 96, 68)
        else:
            raise NotImplementedError("The extraction of hippocampi was not implemented for "
                                      "preprocessing %s" % self.preprocessing)
        crop_size = (50, 50, 50)  # the output cropped hippocampus size

        extracted_roi = image_tensor[
            :,
            crop_center[0] - crop_size[0] // 2: crop_center[0] + crop_size[0] // 2:,
            crop_center[1] - crop_size[1] // 2: crop_center[1] + crop_size[1] // 2:,
            crop_center[2] - crop_size[2] // 2: crop_center[2] + crop_size[2] // 2:
        ].clone()

        return extracted_roi


class MRIDatasetSlice(MRIDataset):

    def __init__(self, caps_directory, data_file, preprocessing="t1-linear",
                 transformations=None, mri_plane=0, prepare_dl=False,
                 discarded_slices=20, mixed=False):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            transformations (callable, optional): Optional transform to be applied on a sample.
            prepare_dl (bool): If true pre-extracted patches will be loaded.
            mri_plane (int): Defines which mri plane is used for slice extraction.
            discarded_slices (int or list): number of slices discarded at the beginning and the end of the image.
                If one single value is given, the same amount is discarded at the beginning and at the end.
            mixed (bool): If True will look for a 'slice_id' column in the input DataFrame to load each slice
                independently.
        """
        # Rename MRI plane
        self.mri_plane = mri_plane
        self.direction_list = ['sag', 'cor', 'axi']
        if self.mri_plane >= len(self.direction_list):
            raise ValueError(
                "mri_plane value %i > %i" %
                (self.mri_plane, len(
                    self.direction_list)))

        # Manage discarded_slices
        if isinstance(discarded_slices, int):
            discarded_slices = [discarded_slices, discarded_slices]
        if isinstance(discarded_slices, list) and len(discarded_slices) == 1:
            discarded_slices = discarded_slices * 2
        self.discarded_slices = discarded_slices

        if mixed:
            self.elem_index = "mixed"
        else:
            self.elem_index = None

        self.mode = "slice"
        super().__init__(caps_directory, data_file, preprocessing, transformations)
        self.prepare_dl = prepare_dl

    def __getitem__(self, idx):
        participant, session, slice_idx, label = self._get_meta_data(idx)
        slice_idx = slice_idx + self.discarded_slices[0]

        if self.prepare_dl:
            # read the slices directly
            slice_path = path.join(self._get_path(participant, session, "slice")[0:-7]
                                   + '_axis-%s' % self.direction_list[self.mri_plane]
                                   + '_channel-rgb_slice-%i_T1w.pt' % slice_idx)
            image = torch.load(slice_path)
        else:
            image_path = self._get_path(participant, session, "image")
            full_image = torch.load(image_path)
            image = self.extract_slice_from_mri(full_image, slice_idx)

        if self.transformations:
            image = self.transformations(image)

        sample = {'image': image, 'label': label,
                  'participant_id': participant, 'session_id': session,
                  'slice_id': slice_idx}

        return sample

    def num_elem_per_image(self):
        if self.elem_index == "mixed":
            return 1

        image = self._get_full_image()
        return image.size(self.mri_plane + 1) - \
            self.discarded_slices[0] - self.discarded_slices[1]

    def extract_slice_from_mri(self, image, index_slice):
        """
        This is a function to grab one slice in each view and create a rgb image for transferring learning: duplicate the slices into R, G, B channel
        :param image: (tensor)
        :param index_slice: (int) index of the wanted slice
        :return:
        To note, for each view:
        Axial_view = "[:, :, slice_i]"
        Coronal_view = "[:, slice_i, :]"
        Sagittal_view= "[slice_i, :, :]"
        """
        image = image.squeeze(0)
        simple_slice = image[(slice(None),) * self.mri_plane + (index_slice,)]
        triple_slice = torch.stack((simple_slice, simple_slice, simple_slice))

        return triple_slice


def return_dataset(mode, input_dir, data_df, preprocessing,
                   transformations, params, cnn_index=None):
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

    slice_participant_list = [
        ele for ele in participant_list for _ in range(slices_per_patient)]
    slice_session_list = [
        ele for ele in session_list for _ in range(slices_per_patient)]
    slice_label_list = [
        ele for ele in label_list for _ in range(slices_per_patient)]
    slice_index_list = slice_index * len(label_list)

    df_final = pd.DataFrame(
        columns=[
            'participant_id',
            'session_id',
            'slice_id',
            'diagnosis'])
    df_final['participant_id'] = np.array(slice_participant_list)
    df_final['session_id'] = np.array(slice_session_list)
    df_final['slice_id'] = np.array(slice_index_list)
    df_final['diagnosis'] = np.array(slice_label_list)

    y = np.array(slice_label_list)
    # split the train data into training and validation set
    skf_2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=10000)
    indices = next(skf_2.split(np.zeros(len(y)), y))
    train_ind, valid_ind = indices

    df_sub_train = df_final.iloc[train_ind]
    df_sub_valid = df_final.iloc[valid_ind]

    df_sub_train.reset_index(inplace=True, drop=True)
    df_sub_valid.reset_index(inplace=True, drop=True)

    return df_sub_train, df_sub_valid
