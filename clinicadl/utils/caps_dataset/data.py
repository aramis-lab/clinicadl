# coding: utf8

import abc
from logging import getLogger
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from clinica.utils.exceptions import ClinicaCAPSError
from torch.utils.data import Dataset

from clinicadl.extract.extract_utils import (
    PATTERN_DICT,
    TEMPLATE_DICT,
    compute_discarded_slices,
    compute_folder_and_file_type,
    extract_patch_path,
    extract_patch_tensor,
    extract_roi_path,
    extract_roi_tensor,
    extract_slice_path,
    extract_slice_tensor,
    find_mask_path,
)

logger = getLogger("clinicadl")


#################################
# Datasets loaders
#################################
class CapsDataset(Dataset):
    """Abstract class for all derived CapsDatasets."""

    def __init__(
        self,
        caps_directory: str,
        data_df: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        transformations: Optional[Callable],
        label_presence: bool,
        label: str = None,
        label_code: Dict[Any, int] = None,
        augmentation_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
    ):
        self.caps_directory = caps_directory
        self.caps_dict = self.create_caps_dict(caps_directory, multi_cohort)
        self.transformations = transformations
        self.augmentation_transformations = augmentation_transformations
        self.eval_mode = False
        self.label_presence = label_presence
        self.label = label
        self.label_code = label_code
        self.preprocessing_dict = preprocessing_dict

        if not hasattr(self, "elem_index"):
            raise ValueError(
                "Child class of CapsDataset must set elem_index attribute."
            )
        if not hasattr(self, "mode"):
            raise ValueError("Child class of CapsDataset must set mode attribute.")

        self.df = data_df

        mandatory_col = {"participant_id", "session_id", "cohort"}
        if self.label_presence and self.label is not None:
            mandatory_col.add(self.label)

        if not mandatory_col.issubset(set(self.df.columns.values)):
            raise Exception(
                "the data file is not in the correct format."
                "Columns should include %s" % mandatory_col
            )

        self.elem_per_image = self.num_elem_per_image()
        self.size = self[0]["image"].size()

    @property
    @abc.abstractmethod
    def elem_index(self):
        pass

    def label_fn(self, target: Union[str, float, int]) -> Union[float, int]:
        """
        Returns the label value usable in criterion.

        Args:
            target: value of the target.
        Returns:
            label: value of the label usable in criterion.
        """
        # Reconstruction case (no label)
        if self.label is None:
            return None
        # Regression case (no label code)
        elif self.label_code is None:
            return np.float32([target])
        # Classification case (label + label_code dict)
        else:
            return self.label_code[str(target)]

    def __len__(self) -> int:
        return len(self.df) * self.elem_per_image

    @staticmethod
    def create_caps_dict(caps_directory: str, multi_cohort: bool) -> Dict[str, str]:

        from clinica.utils.inputs import check_caps_folder

        if multi_cohort:
            if not caps_directory.endswith(".tsv"):
                raise ValueError(
                    "If multi_cohort is given, the caps_dir argument should be a path to a TSV file."
                )
            else:
                caps_df = pd.read_csv(caps_directory, sep="\t")
                check_multi_cohort_tsv(caps_df, "CAPS")
                caps_dict = dict()
                for idx in range(len(caps_df)):
                    cohort = caps_df.loc[idx, "cohort"]
                    caps_path = caps_df.loc[idx, "path"]
                    check_caps_folder(caps_path)
                    caps_dict[cohort] = caps_path
        else:
            check_caps_folder(caps_directory)
            caps_dict = {"single": caps_directory}

        return caps_dict

    def _get_image_path(self, participant: str, session: str, cohort: str) -> str:
        """
        Gets the path to the tensor image (*.pt)

        Args:
            participant: ID of the participant.
            session: ID of the session.
            cohort: Name of the cohort.
        Returns:
            image_path: path to the tensor containing the whole image.
        """
        from clinica.utils.inputs import clinica_file_reader

        # Try to find .nii.gz file
        try:
            file_type = self.preprocessing_dict["file_type"]
            results = clinica_file_reader(
                [participant], [session], self.caps_dict[cohort], file_type
            )
            image_filename = path.basename(results[0]).replace(".nii.gz", ".pt")
            folder, _ = compute_folder_and_file_type(self.preprocessing_dict)
            image_dir = path.join(
                self.caps_dict[cohort],
                "subjects",
                participant,
                session,
                "deeplearning_prepare_data",
                "image_based",
                folder,
            )
            image_path = path.join(image_dir, image_filename)
        # Try to find .pt file
        except ClinicaCAPSError:
            file_type = self.preprocessing_dict["file_type"]
            file_type["pattern"] = file_type["pattern"].replace(".nii.gz", ".pt")
            results = clinica_file_reader(
                [participant], [session], self.caps_dict[cohort], file_type
            )
            image_path = results[0]

        return image_path

    def _get_meta_data(self, idx: int) -> Tuple[str, str, str, int, int]:
        """
        Gets all meta data necessary to compute the path with _get_image_path

        Args:
            idx (int): row number of the meta-data contained in self.df
        Returns:
            participant (str): ID of the participant.
            session (str): ID of the session.
            cohort (str): Name of the cohort.
            elem_index (int): Index of the part of the image.
            label (str or float or int): value of the label to be used in criterion.
        """
        image_idx = idx // self.elem_per_image
        participant = self.df.loc[image_idx, "participant_id"]
        session = self.df.loc[image_idx, "session_id"]
        cohort = self.df.loc[image_idx, "cohort"]

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        else:
            elem_idx = self.elem_index

        if self.label_presence and self.label is not None:
            target = self.df.loc[image_idx, self.label]
            label = self.label_fn(target)
        else:
            label = -1

        return participant, session, cohort, elem_idx, label

    def _get_full_image(self) -> torch.Tensor:
        """
        Allows to get the an example of the image mode corresponding to the dataset.
        Useful to compute the number of elements if mode != image.

        Returns:
            image tensor of the full image first image.
        """
        import nibabel as nib
        from clinica.utils.inputs import clinica_file_reader

        participant_id = self.df.loc[0, "participant_id"]
        session_id = self.df.loc[0, "session_id"]
        cohort = self.df.loc[0, "cohort"]

        try:
            image_path = self._get_image_path(participant_id, session_id, cohort)
            image = torch.load(image_path)
        except IndexError:
            file_type = self.preprocessing_dict["file_type"]
            results = clinica_file_reader(
                [participant_id], [session_id], self.caps_dict[cohort], file_type
            )
            image_nii = nib.load(results[0])
            image_np = image_nii.get_fdata()
            image = ToTensor()(image_np)

        return image

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Gets the sample containing all the information needed for training and testing tasks.

        Args:
            idx: row number of the meta-data contained in self.df
        Returns:
            dictionary with following items:
                - "image" (torch.Tensor): the input given to the model,
                - "label" (int or float): the label used in criterion,
                - "participant_id" (str): ID of the participant,
                - "session_id" (str): ID of the session,
                - f"{self.mode}_id" (int): number of the element,
                - "image_path": path to the image loaded in CAPS.

        """
        pass

    @abc.abstractmethod
    def num_elem_per_image(self) -> int:
        """Computes the number of elements per image based on the full image."""
        pass

    def eval(self):
        """Put the dataset on evaluation mode (data augmentation is not performed)."""
        self.eval_mode = True
        return self

    def train(self):
        """Put the dataset on training mode (data augmentation is performed)."""
        self.eval_mode = False
        return self


class CapsDatasetImage(CapsDataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(
        self,
        caps_directory: str,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        train_transformations: Optional[Callable] = None,
        label_presence: bool = True,
        label: str = None,
        label_code: Dict[str, int] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing_dict: preprocessing dict contained in the JSON file of extract.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

        """
        self.mode = "image"
        super().__init__(
            caps_directory,
            data_file,
            preprocessing_dict,
            augmentation_transformations=train_transformations,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            transformations=all_transformations,
            multi_cohort=multi_cohort,
        )

    @property
    def elem_index(self):
        return None

    def __getitem__(self, idx):
        participant, session, cohort, _, label = self._get_meta_data(idx)

        image_path = self._get_image_path(participant, session, cohort)
        image = torch.load(image_path)

        if self.transformations:
            image = self.transformations(image)

        if self.augmentation_transformations and not self.eval_mode:
            image = self.augmentation_transformations(image)

        sample = {
            "image": image,
            "label": label,
            "participant_id": participant,
            "session_id": session,
            "image_id": 0,
            "image_path": image_path,
        }

        return sample

    def num_elem_per_image(self):
        return 1


class CapsDatasetPatch(CapsDataset):
    def __init__(
        self,
        caps_directory: str,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        train_transformations: Optional[Callable] = None,
        patch_index: Optional[int] = None,
        label_presence: bool = True,
        label: str = None,
        label_code: Dict[str, int] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing_dict: preprocessing dict contained in the JSON file of extract.
            train_transformations: Optional transform to be applied only on training mode.
            patch_index: If a value is given the same patch location will be extracted for each image.
                else the dataset will load all the patches possible for one image.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

        """
        self.patch_size = preprocessing_dict["patch_size"]
        self.stride_size = preprocessing_dict["stride_size"]
        self.patch_index = patch_index
        self.mode = "patch"
        self.prepare_dl = preprocessing_dict["prepare_dl"]
        super().__init__(
            caps_directory,
            data_file,
            preprocessing_dict,
            augmentation_transformations=train_transformations,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            transformations=all_transformations,
            multi_cohort=multi_cohort,
        )

    @property
    def elem_index(self):
        return self.patch_index

    def __getitem__(self, idx):
        participant, session, cohort, patch_idx, label = self._get_meta_data(idx)
        image_path = self._get_image_path(participant, session, cohort)

        if self.prepare_dl:
            patch_dir = path.dirname(image_path).replace(
                "image_based", f"{self.mode}_based"
            )
            patch_filename = extract_patch_path(
                image_path, self.patch_size, self.stride_size, patch_idx
            )
            patch_tensor = torch.load(path.join(patch_dir, patch_filename))

        else:
            image = torch.load(image_path)
            patch_tensor = extract_patch_tensor(
                image, self.patch_size, self.stride_size, patch_idx
            )

        if self.transformations:
            patch_tensor = self.transformations(patch_tensor)

        if self.augmentation_transformations and not self.eval_mode:
            patch_tensor = self.augmentation_transformations(patch_tensor)

        sample = {
            "image": patch_tensor,
            "label": label,
            "participant_id": participant,
            "session_id": session,
            "patch_id": patch_idx,
        }

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1

        image = self._get_full_image()

        patches_tensor = (
            image.unfold(1, self.patch_size, self.stride_size)
            .unfold(2, self.patch_size, self.stride_size)
            .unfold(3, self.patch_size, self.stride_size)
            .contiguous()
        )
        patches_tensor = patches_tensor.view(
            -1, self.patch_size, self.patch_size, self.patch_size
        )
        num_patches = patches_tensor.shape[0]
        return num_patches


class CapsDatasetRoi(CapsDataset):
    def __init__(
        self,
        caps_directory: str,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        roi_index: Optional[int] = None,
        train_transformations: Optional[Callable] = None,
        label_presence: bool = True,
        label: str = None,
        label_code: Dict[str, int] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing_dict: preprocessing dict contained in the JSON file of extract.
            roi_index: If a value is given the same region will be extracted for each image.
                else the dataset will load all the regions possible for one image.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

        """
        self.roi_index = roi_index
        self.mode = "roi"
        self.roi_list = preprocessing_dict["roi_list"]
        self.uncropped_roi = preprocessing_dict["uncropped_roi"]
        self.prepare_dl = preprocessing_dict["prepare_dl"]
        self.mask_paths, self.mask_arrays = self._get_mask_paths_and_tensors(
            caps_directory, preprocessing_dict
        )
        super().__init__(
            caps_directory,
            data_file,
            preprocessing_dict,
            augmentation_transformations=train_transformations,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            transformations=all_transformations,
            multi_cohort=multi_cohort,
        )

    @property
    def elem_index(self):
        return self.roi_index

    def __getitem__(self, idx):
        participant, session, cohort, roi_idx, label = self._get_meta_data(idx)
        image_path = self._get_image_path(participant, session, cohort)

        if self.roi_list is None:
            raise NotImplementedError(
                "Default regions are not available anymore in ClinicaDL. "
                "Please define appropriate masks and give a roi_list."
            )

        if self.prepare_dl:
            mask_path = self.mask_paths[roi_idx]
            roi_dir = path.dirname(image_path).replace(
                "image_based", f"{self.mode}_based"
            )
            roi_filename = extract_roi_path(image_path, mask_path, self.uncropped_roi)
            roi_tensor = torch.load(path.join(roi_dir, roi_filename))

        else:
            image = torch.load(image_path)
            mask_array = self.mask_arrays[roi_idx]
            roi_tensor = extract_roi_tensor(image, mask_array, self.uncropped_roi)

        if self.transformations:
            roi_tensor = self.transformations(roi_tensor)

        if self.augmentation_transformations and not self.eval_mode:
            roi_tensor = self.augmentation_transformations(roi_tensor)

        sample = {
            "image": roi_tensor,
            "label": label,
            "participant_id": participant,
            "session_id": session,
            "roi_id": roi_idx,
        }

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1
        if self.roi_list is None:
            return 2
        else:
            return len(self.roi_list)

    def _get_mask_paths_and_tensors(
        self, caps_directory: str, preprocessing_dict: Dict[str, Any]
    ) -> Tuple[List[str], List]:
        """Loads the masks necessary to regions extraction"""
        import nibabel as nib

        # Find template name
        if preprocessing_dict["preprocessing"] == "custom":
            template_name = preprocessing_dict["roi_custom_template"]
            if template_name is None:
                raise ValueError(
                    f"Please provide a name for the template when preprocessing is `custom`."
                )
        elif preprocessing_dict["preprocessing"] in TEMPLATE_DICT:
            template_name = TEMPLATE_DICT[preprocessing_dict["preprocessing"]]
        else:
            raise NotImplementedError(
                f"Template of preprocessing {preprocessing_dict['preprocessing']} "
                f"is not defined."
            )

        # Find mask pattern
        if preprocessing_dict["preprocessing"] == "custom":
            pattern = preprocessing_dict["roi_custom_mask_pattern"]
            if pattern is None:
                raise ValueError(
                    f"Please provide a pattern for the masks when preprocessing is `custom`."
                )
        elif preprocessing_dict["preprocessing"] in PATTERN_DICT:
            pattern = PATTERN_DICT[preprocessing_dict["preprocessing"]]
        else:
            raise NotImplementedError(
                f"Pattern of mask for preprocessing {preprocessing_dict['preprocessing']} "
                f"is not defined."
            )

        mask_location = path.join(caps_directory, "masks", f"tpl-{template_name}")

        mask_paths, mask_arrays = list(), list()
        for roi in self.roi_list:
            logger.info(f"Find mask for roi {roi}.")
            mask_path, desc = find_mask_path(mask_location, roi, pattern, True)
            if mask_path is None:
                raise ValueError(desc)
            mask_nii = nib.load(mask_path)
            mask_paths.append(mask_path)
            mask_arrays.append(mask_nii.get_fdata())

        return mask_paths, mask_arrays


class CapsDatasetSlice(CapsDataset):
    def __init__(
        self,
        caps_directory: str,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        slice_index: Optional[int] = None,
        train_transformations: Optional[Callable] = None,
        label_presence: bool = True,
        label: str = None,
        label_code: Dict[str, int] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing_dict: preprocessing dict contained in the JSON file of extract.
            slice_index: If a value is given the same slice will be extracted for each image.
                else the dataset will load all the slices possible for one image.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.
        """
        self.slice_index = slice_index
        self.slice_direction = preprocessing_dict["slice_direction"]
        self.slice_mode = preprocessing_dict["slice_mode"]
        self.discarded_slices = compute_discarded_slices(
            preprocessing_dict["discarded_slices"]
        )
        self.num_slices = None
        if "num_slices" in preprocessing_dict:
            self.num_slices = preprocessing_dict["num_slices"]

        self.mode = "slice"
        self.prepare_dl = preprocessing_dict["prepare_dl"]
        super().__init__(
            caps_directory,
            data_file,
            preprocessing_dict,
            augmentation_transformations=train_transformations,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            transformations=all_transformations,
            multi_cohort=multi_cohort,
        )

    @property
    def elem_index(self):
        return self.slice_index

    def __getitem__(self, idx):
        participant, session, cohort, slice_idx, label = self._get_meta_data(idx)
        slice_idx = slice_idx + self.discarded_slices[0]
        image_path = self._get_image_path(participant, session, cohort)

        if self.prepare_dl:
            slice_dir = path.dirname(image_path).replace(
                "image_based", f"{self.mode}_based"
            )
            slice_filename = extract_slice_path(
                image_path, self.slice_direction, self.slice_mode, slice_idx
            )
            slice_tensor = torch.load(path.join(slice_dir, slice_filename))

        else:
            image_path = self._get_image_path(participant, session, cohort)
            image = torch.load(image_path)
            slice_tensor = extract_slice_tensor(
                image, self.slice_direction, self.slice_mode, slice_idx
            )

        if self.transformations:
            slice_tensor = self.transformations(slice_tensor)

        if self.augmentation_transformations and not self.eval_mode:
            slice_tensor = self.augmentation_transformations(slice_tensor)

        sample = {
            "image": slice_tensor,
            "label": label,
            "participant_id": participant,
            "session_id": session,
            "slice_id": slice_idx,
        }

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1

        if self.num_slices is not None:
            return self.num_slices

        image = self._get_full_image()
        return (
            image.size(self.slice_direction + 1)
            - self.discarded_slices[0]
            - self.discarded_slices[1]
        )


def return_dataset(
    input_dir: str,
    data_df: pd.DataFrame,
    preprocessing_dict: Dict[str, Any],
    all_transformations: Optional[Callable],
    label: str = None,
    label_code: Dict[str, int] = None,
    train_transformations: Optional[Callable] = None,
    cnn_index: int = None,
    label_presence: bool = True,
    multi_cohort: bool = False,
) -> CapsDataset:
    """
    Return appropriate Dataset according to given options.
    Args:
        input_dir: path to a directory containing a CAPS structure.
        data_df: List subjects, sessions and diagnoses.
        preprocessing_dict: preprocessing dict contained in the JSON file of extract.
        train_transformations: Optional transform to be applied during training only.
        all_transformations: Optional transform to be applied during training and evaluation.
        label: Name of the column in data_df containing the label.
        label_code: label code that links the output node number to label value.
        cnn_index: Index of the CNN in a multi-CNN paradigm (optional).
        label_presence: If True the diagnosis will be extracted from the given DataFrame.
        multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

    Returns:
         the corresponding dataset.
    """

    if cnn_index is not None and preprocessing_dict["mode"] == "image":
        raise ValueError(
            f"Multi-CNN is not implemented for {preprocessing_dict['mode']} mode."
        )

    if preprocessing_dict["mode"] == "image":
        return CapsDatasetImage(
            input_dir,
            data_df,
            preprocessing_dict,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    elif preprocessing_dict["mode"] == "patch":
        return CapsDatasetPatch(
            input_dir,
            data_df,
            preprocessing_dict,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            patch_index=cnn_index,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    elif preprocessing_dict["mode"] == "roi":
        return CapsDatasetRoi(
            input_dir,
            data_df,
            preprocessing_dict,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            roi_index=cnn_index,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    elif preprocessing_dict["mode"] == "slice":
        return CapsDatasetSlice(
            input_dir,
            data_df,
            preprocessing_dict,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            slice_index=cnn_index,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    else:
        raise ValueError(f"Mode {preprocessing_dict['mode']} is not implemented.")


##################################
# Transformations
##################################


class RandomNoising(object):
    """Applies a random zoom to a tensor"""

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, image):
        import random

        sigma = random.uniform(0, self.sigma)
        dist = torch.distributions.normal.Normal(0, sigma)
        return image + dist.sample(image.shape)


class RandomSmoothing(object):
    """Applies a random zoom to a tensor"""

    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, image):
        import random

        from scipy.ndimage import gaussian_filter

        sigma = random.uniform(0, self.sigma)
        image = gaussian_filter(image, sigma)  # smoothing of data
        image = torch.from_numpy(image).float()
        return image


class RandomCropPad(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, image):
        dimensions = len(image.shape) - 1
        crop = np.random.randint(-self.length, self.length, dimensions)
        if dimensions == 2:
            output = torch.nn.functional.pad(
                image, (-crop[0], crop[0], -crop[1], crop[1])
            )
        elif dimensions == 3:
            output = torch.nn.functional.pad(
                image, (-crop[0], crop[0], -crop[1], crop[1], -crop[2], crop[2])
            )
        else:
            raise ValueError("RandomCropPad is only available for 2D or 3D data.")
        return output


class GaussianSmoothing(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        from scipy.ndimage.filters import gaussian_filter

        image = sample["image"]
        np.nan_to_num(image, copy=False)
        smoothed_image = gaussian_filter(image, sigma=self.sigma)
        sample["image"] = smoothed_image

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


def get_transforms(
    minmaxnormalization: bool = True, data_augmentation: List[str] = None
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Outputs the transformations that will be applied to the dataset

    Args:
        minmaxnormalization: if True will perform MinMaxNormalization.
        data_augmentation: list of data augmentation performed on the training set.

    Returns:
        transforms to apply in train and evaluation mode / transforms to apply in evaluation mode only.
    """
    augmentation_dict = {
        "Noise": RandomNoising(sigma=0.1),
        "Erasing": transforms.RandomErasing(),
        "CropPad": RandomCropPad(10),
        "Smoothing": RandomSmoothing(),
        "None": None,
    }
    if data_augmentation:
        augmentation_list = [
            augmentation_dict[augmentation] for augmentation in data_augmentation
        ]
    else:
        augmentation_list = []

    if minmaxnormalization:
        transformations_list = [MinMaxNormalization()]
    else:
        transformations_list = []

    all_transformations = transforms.Compose(transformations_list)
    train_transformations = transforms.Compose(augmentation_list)

    return train_transformations, all_transformations


################################
# TSV files loaders
################################
def load_data_test(test_path, diagnoses_list, baseline=True, multi_cohort=False):
    """
    Load data not managed by split_manager.

    Args:
        test_path (str): path to the test TSV files / split directory / TSV file for multi-cohort
        diagnoses_list (List[str]): list of the diagnoses wanted in case of split_dir or multi-cohort
        baseline (bool): If True baseline sessions only used (split_dir handling only).
        multi_cohort (bool): If True considers multi-cohort setting.
    """
    # TODO: computes baseline sessions on-the-fly to manager TSV file case

    if multi_cohort:
        if not test_path.endswith(".tsv"):
            raise ValueError(
                "If multi_cohort is given, the tsv_path argument should be a path to a TSV file."
            )
        else:
            tsv_df = pd.read_csv(test_path, sep="\t")
            check_multi_cohort_tsv(tsv_df, "labels")
            test_df = pd.DataFrame()
            found_diagnoses = set()
            for idx in range(len(tsv_df)):
                cohort_name = tsv_df.loc[idx, "cohort"]
                cohort_path = tsv_df.loc[idx, "path"]
                cohort_diagnoses = (
                    tsv_df.loc[idx, "diagnoses"].replace(" ", "").split(",")
                )
                if bool(set(cohort_diagnoses) & set(diagnoses_list)):
                    target_diagnoses = list(set(cohort_diagnoses) & set(diagnoses_list))
                    cohort_test_df = load_data_test_single(
                        cohort_path, target_diagnoses, baseline=baseline
                    )
                    cohort_test_df["cohort"] = cohort_name
                    test_df = pd.concat([test_df, cohort_test_df])
                    found_diagnoses = found_diagnoses | (
                        set(cohort_diagnoses) & set(diagnoses_list)
                    )

            if found_diagnoses != set(diagnoses_list):
                raise ValueError(
                    f"The diagnoses found in the multi cohort dataset {found_diagnoses} "
                    f"do not correspond to the diagnoses wanted {set(diagnoses_list)}."
                )
            test_df.reset_index(inplace=True, drop=True)
    else:
        if test_path.endswith(".tsv"):
            tsv_df = pd.read_csv(test_path, sep="\t")
            multi_col = {"cohort", "path"}
            if multi_col.issubset(tsv_df.columns.values):
                raise ValueError(
                    "To use multi-cohort framework, please add --multi_cohort flag."
                )
        test_df = load_data_test_single(test_path, diagnoses_list, baseline=baseline)
        test_df["cohort"] = "single"

    return test_df


def load_data_test_single(test_path, diagnoses_list, baseline=True):

    if test_path.endswith(".tsv"):
        test_df = pd.read_csv(test_path, sep="\t")
        if "diagnosis" not in test_df.columns.values:
            raise ValueError(
                f"'diagnosis' column must be present in TSV file {test_path}."
            )
        test_df = test_df[test_df.diagnosis.isin(diagnoses_list)]
        if len(test_df) == 0:
            raise ValueError(
                f"Diagnoses wanted {diagnoses_list} were not found in TSV file {test_path}."
            )
        return test_df

    test_df = pd.DataFrame()

    for diagnosis in diagnoses_list:

        if baseline:
            test_diagnosis_path = path.join(test_path, diagnosis + "_baseline.tsv")
        else:
            test_diagnosis_path = path.join(test_path, diagnosis + ".tsv")

        test_diagnosis_df = pd.read_csv(test_diagnosis_path, sep="\t")
        test_df = pd.concat([test_df, test_diagnosis_df])

    test_df.reset_index(inplace=True, drop=True)

    return test_df


def check_multi_cohort_tsv(tsv_df, purpose):
    """
    Checks that a multi-cohort TSV file is valid.

    Args:
        tsv_df (pd.DataFrame): DataFrame of multi-cohort definition.
        purpose (str): what the TSV file describes (CAPS or TSV).
    Raises:
        ValueError: if the TSV file is badly formatted.
    """
    if purpose.upper() == "CAPS":
        mandatory_col = {"cohort", "path"}
    else:
        mandatory_col = {"cohort", "path", "diagnoses"}
    if not mandatory_col.issubset(tsv_df.columns.values):
        raise ValueError(
            f"Columns of the TSV file used for {purpose} location must include {mandatory_col}"
        )
