# coding: utf8
# TODO: create a folder for generate/ prepare_data/ data to deal with capsDataset objects ?
import abc
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.caps_dataset.caps_dataset_utils import compute_folder_and_file_type
from clinicadl.prepare_data.prepare_data_utils import (
    compute_discarded_slices,
    extract_patch_path,
    extract_patch_tensor,
    extract_roi_path,
    extract_roi_tensor,
    extract_slice_path,
    extract_slice_tensor,
    find_mask_path,
)
from clinicadl.utils.enum import (
    ExtractionMethod,
    Pattern,
    Preprocessing,
    SliceDirection,
    SliceMode,
    Template,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
)

logger = getLogger("clinicadl")


#################################
# Datasets loaders
#################################
class CapsDataset(Dataset):
    """Abstract class for all derived CapsDatasets."""

    def __init__(
        self,
        config: CapsDatasetConfig,
    ):
        if not hasattr(self, "elem_index"):
            raise AttributeError(
                "Child class of CapsDataset must set elem_index attribute."
            )
        if not hasattr(self, "mode"):
            raise AttributeError("Child class of CapsDataset, must set mode attribute.")
        self.config = config
        self.df = config.data.data_df
        mandatory_col = {
            "participant_id",
            "session_id",
            "cohort",
        }
        if config.data.label_presence and config.data.label is not None:
            mandatory_col.add(config.data.label)

        if not mandatory_col.issubset(set(self.df.columns.values)):
            raise Exception(
                f"the data file is not in the correct format."
                f"Columns should include {mandatory_col}"
            )
        self.elem_per_image = self.num_elem_per_image()
        self.size = self[0]["image"].size()

    @property
    @abc.abstractmethod
    def elem_index(self):
        pass

    def label_fn(self, target: Union[str, float, int]) -> Union[float, int, None]:
        """
        Returns the label value usable in criterion.

        Args:
            target: value of the target.
        Returns:
            label: value of the label usable in criterion.
        """
        # Reconstruction case (no label)
        if self.config.data.label is None:
            return None
        # Regression case (no label code)
        elif self.config.data.label_code is None:
            return np.float32([target])
        # Classification case (label + label_code dict)
        else:
            return self.config.data.label_code[str(target)]

    def domain_fn(self, target: Union[str, float, int]) -> Union[float, int]:
        """
        Returns the label value usable in criterion.

        Args:
            target: value of the target.
        Returns:
            label: value of the label usable in criterion.
        """
        domain_code = {"t1": 0, "flair": 1}
        return domain_code[str(target)]

    def __len__(self) -> int:
        return len(self.df) * self.elem_per_image

    def _get_image_path(self, participant: str, session: str, cohort: str) -> Path:
        """
        Gets the path to the tensor image (*.pt)

        Args:
            participant: ID of the participant.
            session: ID of the session.
            cohort: Name of the cohort.
        Returns:
            image_path: path to the tensor containing the whole image.
        """
        from clinicadl.utils.clinica_utils import clinicadl_file_reader

        # Try to find .nii.gz file
        try:
            file_type = self.config.preprocessing.file_type
            results = clinicadl_file_reader(
                [participant], [session], self.config.data.caps_dict[cohort], file_type
            )
            logger.debug(f"clinicadl_file_reader output: {results}")
            filepath = Path(results[0][0])
            image_filename = filepath.name.replace(".nii.gz", ".pt")

            folder, _ = compute_folder_and_file_type(self.config)
            image_dir = (
                self.config.data.caps_dict[cohort]
                / "subjects"
                / participant
                / session
                / "deeplearning_prepare_data"
                / "image_based"
                / folder
            )
            image_path = image_dir / image_filename
        # Try to find .pt file
        except ClinicaDLCAPSError:
            file_type = self.config.preprocessing.file_type
            file_type["pattern"] = file_type["pattern"].replace(".nii.gz", ".pt")
            results = clinicadl_file_reader(
                [participant], [session], self.config.data.caps_dict[cohort], file_type
            )
            filepath = results[0]
            image_path = Path(filepath[0])

        return image_path

    def _get_meta_data(
        self, idx: int
    ) -> Tuple[str, str, str, Union[float, int, None], int]:
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
        participant = self.df.at[image_idx, "participant_id"]
        session = self.df.at[image_idx, "session_id"]
        cohort = self.df.at[image_idx, "cohort"]

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        else:
            elem_idx = self.elem_index
        if self.config.data.label_presence and self.config.data.label is not None:
            target = self.df.at[image_idx, self.config.data.label]
            label = self.label_fn(target)
        else:
            label = -1

        if "domain" in self.df.columns:
            domain = self.df.at[image_idx, "domain"]
            domain = self.domain_fn(domain)
        else:
            domain = ""  # TO MODIFY
        return participant, session, cohort, elem_idx, label, domain

    def _get_full_image(self) -> torch.Tensor:
        """
        Allows to get the an example of the image mode corresponding to the dataset.
        Useful to compute the number of elements if mode != image.

        Returns:
            image tensor of the full image first image.
        """
        import nibabel as nib

        from clinicadl.utils.clinica_utils import clinicadl_file_reader

        participant_id = self.df.loc[0, "participant_id"]
        session_id = self.df.loc[0, "session_id"]
        cohort = self.df.loc[0, "cohort"]

        try:
            image_path = self._get_image_path(participant_id, session_id, cohort)
            image = torch.load(image_path)
        except IndexError:
            file_type = self.config.preprocessing.file_type
            results = clinicadl_file_reader(
                [participant_id],
                [session_id],
                self.config.data.caps_dict[cohort],
                file_type,
            )
            image_nii = nib.loadsave.load(results[0])
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
        config: CapsDatasetConfig,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

        """

        self.config = config
        super().__init__(config=config)

    @property
    def elem_index(self):
        return None

    def __getitem__(self, idx):
        participant, session, cohort, _, label, domain = self._get_meta_data(idx)

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
            "image_path": image_path.as_posix(),
            "domain": domain,
        }

        return sample

    def num_elem_per_image(self):
        return 1


class CapsDatasetPatch(CapsDataset):
    def __init__(
        self,
        config: CapsDatasetConfig,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
            train_transformations: Optional transform to be applied only on training mode.
            patch_index: If a value is given the same patch location will be extracted for each image.
                else the dataset will load all the patches possible for one image.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

        """
        self.config = config
        super().__init__(config)

    @property
    def elem_index(self):
        return self.patch_index

    def __getitem__(self, idx):
        participant, session, cohort, patch_idx, label, domain = self._get_meta_data(
            idx
        )
        image_path = self._get_image_path(participant, session, cohort)

        if self.config.preprocessing.save_features:
            patch_dir = image_path.parent.as_posix().replace(
                "image_based", f"{self.mode}_based"
            )
            patch_filename = extract_patch_path(
                image_path,
                self.config.preprocessing.patch_size,
                self.config.preprocessing.stride_size,
                patch_idx,
            )
            patch_tensor = torch.load(Path(patch_dir).resolve() / patch_filename)

        else:
            image = torch.load(image_path)
            patch_tensor = extract_patch_tensor(
                image,
                self.config.preprocessing.patch_size,
                self.config.preprocessing.stride_size,
                patch_idx,
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
            image.unfold(
                1,
                self.config.preprocessing.patch_size,
                self.config.preprocessing.stride_size,
            )
            .unfold(
                2,
                self.config.preprocessing.patch_size,
                self.config.preprocessing.stride_size,
            )
            .unfold(
                3,
                self.config.preprocessing.patch_size,
                self.config.preprocessing.stride_size,
            )
            .contiguous()
        )
        patches_tensor = patches_tensor.view(
            -1,
            self.config.preprocessing.patch_size,
            self.config.preprocessing.patch_size,
            self.config.preprocessing.patch_size,
        )
        num_patches = patches_tensor.shape[0]
        return num_patches


class CapsDatasetRoi(CapsDataset):
    def __init__(
        self,
        config: CapsDatasetConfig,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
            roi_index: If a value is given the same region will be extracted for each image.
                else the dataset will load all the regions possible for one image.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

        """
        self.config = config
        self.mask_paths, self.mask_arrays = self._get_mask_paths_and_tensors()

        super().__init__(config)

    @property
    def elem_index(self):
        return self.config.preprocessing.roi_index

    def __getitem__(self, idx):
        participant, session, cohort, roi_idx, label, domain = self._get_meta_data(idx)
        image_path = self._get_image_path(participant, session, cohort)

        if self.config.preprocessing.roi_list is None:
            raise NotImplementedError(
                "Default regions are not available anymore in ClinicaDL. "
                "Please define appropriate masks and give a roi_list."
            )

        if self.config.preprocessing.save_features:
            mask_path = self.mask_paths
            roi_dir = image_path.parent.as_posix().replace(
                "image_based", f"{self.config.preprocessing.extract_method.value}_based"
            )
            roi_filename = extract_roi_path(
                image_path, mask_path, self.config.preprocessing.uncropped_roi
            )
            roi_tensor = torch.load(Path(roi_dir) / roi_filename)

        else:
            image = torch.load(image_path)
            mask_array = self.mask_arrays[roi_idx]
            roi_tensor = extract_roi_tensor(
                image, mask_array, self.config.preprocessing.uncropped_roi
            )

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
        if self.config.preprocessing.roi_list is None:
            return 2
        else:
            return len(self.config.preprocessing.roi_list)

    def _get_mask_paths_and_tensors(
        self,
    ) -> Tuple[List[str], List]:
        """Loads the masks necessary to regions extraction"""
        import nibabel as nib

        caps_dict = self.config.data.caps_dict

        if len(caps_dict) > 1:
            caps_directory = caps_dict[next(iter(caps_dict))]
            logger.warning(
                f"The equality of masks is not assessed for multi-cohort training. "
                f"The masks stored in {caps_directory} will be used."
            )

        try:
            preprocessing_ = Preprocessing(self.config.preprocessing.preprocessing)
        except NotImplementedError:
            print(
                f"Template of preprocessing {self.config.preprocessing.preprocessing} "
                f"is not defined."
            )
        # Find template name and pattern
        if preprocessing_.value == "custom":
            template_name = self.config.preprocessing.roi_custom_template
            if template_name is None:
                raise ValueError(
                    f"Please provide a name for the template when preprocessing is `custom`."
                )

            pattern = self.config.preprocessing.roi_custom_mask_pattern
            if pattern is None:
                raise ValueError(
                    f"Please provide a pattern for the masks when preprocessing is `custom`."
                )

        else:
            for template_ in Template:
                if preprocessing_.name == template_.name:
                    template_name = template_

            for pattern_ in Pattern:
                if preprocessing_.name == pattern_.name:
                    pattern = pattern_

        mask_location = caps_directory / "masks" / f"tpl-{template_name}"

        mask_paths, mask_arrays = list(), list()
        for roi in self.config.preprocessing.roi_list:
            logger.info(f"Find mask for roi {roi}.")
            mask_path, desc = find_mask_path(mask_location, roi, pattern, True)
            if mask_path is None:
                raise FileNotFoundError(desc)
            mask_nii = nib.loadsave.load(mask_path)
            mask_paths.append(Path(mask_path))
            mask_arrays.append(mask_nii.get_fdata())

        return mask_paths, mask_arrays


class CapsDatasetSlice(CapsDataset):
    def __init__(
        self,
        config: CapsDatasetConfig,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
            slice_index: If a value is given the same slice will be extracted for each image.
                else the dataset will load all the slices possible for one image.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.
        """
        self.config = config
        super().__init__(config)

    @property
    def elem_index(self):
        return self.config.preprocessing.slice_index

    def __getitem__(self, idx):
        participant, session, cohort, slice_idx, label, domain = self._get_meta_data(
            idx
        )
        slice_idx = slice_idx + self.config.preprocessing.discarded_slices[0]
        image_path = self._get_image_path(participant, session, cohort)

        if self.config.preprocessing.save_features:
            slice_dir = image_path.parent.as_posix().replace(
                "image_based", f"{self.mode}_based"
            )
            slice_filename = extract_slice_path(
                image_path,
                self.config.preprocessing.slice_direction,
                self.config.preprocessing.slice_mode,
                slice_idx,
            )
            slice_tensor = torch.load(Path(slice_dir) / slice_filename)

        else:
            image_path = self._get_image_path(participant, session, cohort)
            image = torch.load(image_path)
            slice_tensor = extract_slice_tensor(
                image,
                self.config.preprocessing.slice_direction,
                self.config.preprocessing.slice_mode,
                slice_idx,
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

        if self.config.preprocessing.num_slices is not None:
            return self.config.preprocessing.num_slices

        image = self._get_full_image()
        return (
            image.size(int(self.config.preprocessing.slice_direction) + 1)
            - self.config.preprocessing.discarded_slices[0]
            - self.config.preprocessing.discarded_slices[1]
        )


def return_dataset(
    input_dir: Path,
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
        preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
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
        raise NotImplementedError(
            f"Multi-CNN is not implemented for {preprocessing_dict['mode']} mode."
        )

    config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod(preprocessing_dict["mode"]),
        preprocessing_type=Preprocessing(preprocessing_dict["preprocessing"]),
        preprocessing=Preprocessing(preprocessing_dict["preprocessing"]),
        caps_directory=input_dir,
        transformations=all_transformations,
        augmentation_transformations=train_transformations,
        eval_mode=False,
        label_presence=label_presence,
        label=label,
        label_code=label_code,
        use_uncropped_image=preprocessing_dict["use_uncropped_image"],
        multi_cohort=multi_cohort,
        data_df=data_df,
        save_features=preprocessing_dict["prepare_dl"]
        if "prepare_dl" in preprocessing_dict
        else False,
    )

    if preprocessing_dict["mode"] == "image":
        return CapsDatasetImage(config)

    elif preprocessing_dict["mode"] == "patch":
        config.preprocessing.patch_size = preprocessing_dict["patch_size"]
        config.preprocessing.stride_size = preprocessing_dict["stride_size"]
        return CapsDatasetPatch(config)

    elif preprocessing_dict["mode"] == "roi":
        config.preprocessing.roi_list = preprocessing_dict["roi_list"]
        config.preprocessing.uncropped_roi = preprocessing_dict["uncropped_roi"]
        return CapsDatasetRoi(config)

    elif preprocessing_dict["mode"] == "slice":
        config.preprocessing.slice_direction = SliceDirection(
            str(preprocessing_dict["slice_direction"])
        )
        config.preprocessing.slice_mode = SliceMode(preprocessing_dict["slice_mode"])
        config.preprocessing.discarded_slices = compute_discarded_slices(
            preprocessing_dict["discarded_slices"]
        )
        config.preprocessing.num_slices = (
            None
            if "num_slices" not in preprocessing_dict
            else preprocessing_dict["num_slices"]
        )
        return CapsDatasetSlice(config)

    else:
        raise NotImplementedError(
            f"Mode {preprocessing_dict['mode']} is not implemented."
        )
