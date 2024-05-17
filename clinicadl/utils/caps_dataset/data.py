# coding: utf8

import abc
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from clinicadl.prepare_data.prepare_data_config import PrepareDataConfig
from clinicadl.prepare_data.prepare_data_utils import (
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
from clinicadl.utils.caps_dataset.data_config import DataConfig
from clinicadl.utils.enum import (
    Pattern,
    Preprocessing,
    Template,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)
from clinicadl.utils.mode.mode_config import ModeConfig, return_mode_config
from clinicadl.utils.preprocessing.preprocessing_config import (
    return_preprocessing_config,
)

logger = getLogger("clinicadl")


#################################
# Datasets loaders
#################################
class CapsDataset(Dataset):
    """Abstract class for all derived CapsDatasets."""

    def __init__(
        self,
        caps_directory: Path,
        data_df: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        transformations: Optional[Callable],
        label_presence: bool,
        label: Optional[str] = None,
        label_code: Optional[Dict[Any, int]] = None,
        augmentation_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
    ):
        # self.caps_dict = self.create_caps_dict(caps_directory, multi_cohort)
        # self.transformations = transformations
        # self.augmentation_transformations = augmentation_transformations
        # self.eval_mode = False
        # self.label_presence = label_presence
        # self.preprocessing_dict = preprocessing_dict

        self.preprocessing = return_preprocessing_config(preprocessing_dict)
        self.mode = return_mode_config(self.preprocessing.preprocessing)
        self.data = DataConfig(
            caps_directory=caps_directory,
            label=label,
            label_code=label_code,
            caps_dict=self.create_caps_dict(caps_directory, multi_cohort),
            transformations=transformations,
            augmentation_transformations=augmentation_transformations,
            eval_mode=False,
            label_presence=label_presence,
            data_df=data_df,
        )

        if not hasattr(self, "elem_index"):
            raise AttributeError(
                "Child class of CapsDataset must set elem_index attribute."
            )
        if not hasattr(self, "mode"):
            raise AttributeError("Child class of CapsDataset, must set mode attribute.")

        # self.df = data_df
        mandatory_col = {
            "participant_id",
            "session_id",
            "cohort",
        }
        if self.data.label_presence and self.data.label is not None:
            mandatory_col.add(self.data.label)

        if not mandatory_col.issubset(set(self.data.data_df.columns.values)):
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

    def label_fn(self, target: Union[str, float, int]) -> Union[float, int]:
        """
        Returns the label value usable in criterion.

        Args:
            target: value of the target.
        Returns:
            label: value of the label usable in criterion.
        """
        # Reconstruction case (no label)
        if self.data.label is None:
            return None
        # Regression case (no label code)
        elif self.data.label_code is None:
            return np.float32([target])
        # Classification case (label + label_code dict)
        else:
            return self.data.label_code[str(target)]

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
        return len(self.data.data_df) * self.elem_per_image

    @staticmethod
    def create_caps_dict(caps_directory: Path, multi_cohort: bool) -> Dict[str, Path]:
        from clinicadl.utils.clinica_utils import check_caps_folder

        if multi_cohort:
            if not caps_directory.suffix == ".tsv":
                raise ClinicaDLArgumentError(
                    "If multi_cohort is True, the CAPS_DIRECTORY argument should be a path to a TSV file."
                )
            else:
                caps_df = pd.read_csv(caps_directory, sep="\t")
                check_multi_cohort_tsv(caps_df, "CAPS")
                caps_dict = dict()
                for idx in range(len(caps_df)):
                    cohort = caps_df.loc[idx, "cohort"]
                    caps_path = Path(caps_df.loc[idx, "path"])
                    check_caps_folder(caps_path)
                    caps_dict[cohort] = caps_path
        else:
            check_caps_folder(caps_directory)
            caps_dict = {"single": caps_directory}

        return caps_dict

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
            file_type = self.preprocessing.file_type
            results = clinicadl_file_reader(
                [participant], [session], self.data.caps_dict[cohort], file_type
            )
            logger.debug(f"clinicadl_file_reader output: {results}")
            filepath = Path(results[0][0])
            image_filename = filepath.name.replace(".nii.gz", ".pt")

            folder, _ = compute_folder_and_file_type(
                PrepareDataConfig(self.preprocessing, self.mode, self.data)
            )
            image_dir = (
                self.data.caps_dict[cohort]
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
            file_type = self.preprocessing.file_type
            file_type["pattern"] = file_type["pattern"].replace(".nii.gz", ".pt")
            results = clinicadl_file_reader(
                [participant], [session], self.data.caps_dict[cohort], file_type
            )
            filepath = results[0]
            image_path = Path(filepath[0])

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
        participant = self.data.data_df.loc[image_idx, "participant_id"]
        session = self.data.data_df.loc[image_idx, "session_id"]
        cohort = self.data.data_df.loc[image_idx, "cohort"]

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        else:
            elem_idx = self.elem_index
        if self.data.label_presence and self.data.label is not None:
            target = self.data.data_df.loc[image_idx, self.data.label]
            label = self.label_fn(target)
        else:
            label = -1

        if "domain" in self.data.data_df.columns:
            domain = self.data.data_df.loc[image_idx, "domain"]
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

        participant_id = self.data.data_df.loc[0, "participant_id"]
        session_id = self.data.data_df.loc[0, "session_id"]
        cohort = self.data.data_df.loc[0, "cohort"]

        try:
            image_path = self._get_image_path(participant_id, session_id, cohort)
            image = torch.load(image_path)
        except IndexError:
            file_type = self.preprocessing.file_type
            results = clinicadl_file_reader(
                [participant_id], [session_id], self.data.caps_dict[cohort], file_type
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
        caps_directory: Path,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        train_transformations: Optional[Callable] = None,
        label_presence: bool = True,
        label: Optional[str] = None,
        label_code: Optional[Dict[str, int]] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
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

        # self.mode = "image"
        # self.prepare_dl = preprocessing_dict["prepare_dl"]
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
        # self.config = PrepareDataImageConfig(
        #     caps_directory=caps_directory,
        #     preprocessing_cls=Preprocessing(preprocessing_dict["preprocessing"]),
        #     use_uncropped_image=preprocessing_dict["use_uncropped_image"],
        # )

    @property
    def elem_index(self):
        return None

    def __getitem__(self, idx):
        participant, session, cohort, _, label, domain = self._get_meta_data(idx)

        image_path = self._get_image_path(participant, session, cohort)
        image = torch.load(image_path)

        if self.data.transformations:
            image = self.data.transformations(image)

        if self.data.augmentation_transformations and not self.eval_mode:
            image = self.data.augmentation_transformations(image)

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
        caps_directory: Path,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        train_transformations: Optional[Callable] = None,
        patch_index: Optional[int] = None,
        label_presence: bool = True,
        label: Optional[str] = None,
        label_code: Optional[Dict[str, int]] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
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

        self.patch_index = patch_index

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
        participant, session, cohort, patch_idx, label, domain = self._get_meta_data(
            idx
        )
        image_path = self._get_image_path(participant, session, cohort)

        if self.preprocessing.save_features:
            patch_dir = image_path.parent.as_posix().replace(
                "image_based", f"{self.preprocessing.extract_method.value}_based"
            )
            patch_filename = extract_patch_path(
                image_path,
                self.preprocessing.patch_size,
                self.preprocessing.stride_size,
                patch_idx,
            )
            patch_tensor = torch.load(Path(patch_dir).resolve() / patch_filename)

        else:
            image = torch.load(image_path)
            patch_tensor = extract_patch_tensor(
                image,
                self.preprocessing.patch_size,
                self.preprocessing.stride_size,
                patch_idx,
            )

        if self.data.transformations:
            patch_tensor = self.data.transformations(patch_tensor)

        if self.data.augmentation_transformations and not self.eval_mode:
            patch_tensor = self.data.augmentation_transformations(patch_tensor)

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
                1, self.preprocessing.patch_size, self.preprocessing.stride_size
            )
            .unfold(2, self.preprocessing.patch_size, self.preprocessing.stride_size)
            .unfold(3, self.preprocessing.patch_size, self.preprocessing.stride_size)
            .contiguous()
        )
        patches_tensor = patches_tensor.view(
            -1,
            self.preprocessing.patch_size,
            self.preprocessing.patch_size,
            self.preprocessing.patch_size,
        )
        num_patches = patches_tensor.shape[0]
        return num_patches


class CapsDatasetRoi(CapsDataset):
    def __init__(
        self,
        caps_directory: Path,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        roi_index: Optional[int] = None,
        train_transformations: Optional[Callable] = None,
        label_presence: bool = True,
        label: Optional[str] = None,
        label_code: Optional[Dict[str, int]] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
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
        self.roi_index = roi_index
        self.mask_paths, self.mask_arrays = self._get_mask_paths_and_tensors(
            caps_directory, multi_cohort, preprocessing_dict
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
        participant, session, cohort, roi_idx, label, domain = self._get_meta_data(idx)
        image_path = self._get_image_path(participant, session, cohort)

        if self.preprocessing.roi_list is None:
            raise NotImplementedError(
                "Default regions are not available anymore in ClinicaDL. "
                "Please define appropriate masks and give a roi_list."
            )

        if self.preprocessing.save_features:
            mask_path = self.mask_paths[roi_idx]
            roi_dir = image_path.parent.as_posix().replace(
                "image_based", f"{self.mode}_based"
            )
            roi_filename = extract_roi_path(
                image_path, mask_path, self.preprocessing.roi_uncrop_output
            )
            roi_tensor = torch.load(Path(roi_dir) / roi_filename)

        else:
            image = torch.load(image_path)
            mask_array = self.mask_arrays[roi_idx]
            roi_tensor = extract_roi_tensor(
                image, mask_array, self.preprocessing.roi_uncrop_output
            )

        if self.data.transformations:
            roi_tensor = self.data.transformations(roi_tensor)

        if self.data.augmentation_transformations and not self.eval_mode:
            roi_tensor = self.data.augmentation_transformations(roi_tensor)

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
        if self.preprocessing.roi_list is None:
            return 2
        else:
            return len(self.preprocessing.roi_list)

    def _get_mask_paths_and_tensors(
        self,
        caps_directory: Path,
        multi_cohort: bool,
        preprocessing_dict: Dict[str, Any],
    ) -> Tuple[List[str], List]:
        """Loads the masks necessary to regions extraction"""
        import nibabel as nib

        caps_dict = self.create_caps_dict(caps_directory, multi_cohort)

        if len(caps_dict) > 1:
            caps_directory = caps_dict[next(iter(caps_dict))]
            logger.warning(
                f"The equality of masks is not assessed for multi-cohort training. "
                f"The masks stored in {caps_directory} will be used."
            )

        try:
            preprocessing_ = Preprocessing(preprocessing_dict["preprocessing"])
        except NotImplementedError:
            print(
                f"Template of preprocessing {preprocessing_dict['preprocessing']} "
                f"is not defined."
            )
        # Find template name and pattern
        if self.preprocessing.preprocessing.value == "custom":
            template_name = self.preprocessing.roi_custom_template
            if template_name is None:
                raise ValueError(
                    f"Please provide a name for the template when preprocessing is `custom`."
                )

            pattern = self.preprocessing.roi_custom_mask_pattern
            if pattern is None:
                raise ValueError(
                    f"Please provide a pattern for the masks when preprocessing is `custom`."
                )

        else:
            for template_ in Template:
                if self.preprocessing.preprocessing.name == template_.name:
                    template_name = template_

            for pattern_ in Pattern:
                if self.preprocessing.preprocessing.name == pattern_.name:
                    pattern = pattern_

        mask_location = caps_directory / "masks" / f"tpl-{template_name}"

        mask_paths, mask_arrays = list(), list()
        for roi in self.preprocessing.roi_list:
            logger.info(f"Find mask for roi {roi}.")
            mask_path, desc = find_mask_path(mask_location, roi, pattern, True)
            if mask_path is None:
                raise FileNotFoundError(desc)
            mask_nii = nib.load(mask_path)
            mask_paths.append(Path(mask_path))
            mask_arrays.append(mask_nii.get_fdata())

        return mask_paths, mask_arrays


class CapsDatasetSlice(CapsDataset):
    def __init__(
        self,
        caps_directory: Path,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        slice_index: Optional[int] = None,
        train_transformations: Optional[Callable] = None,
        label_presence: bool = True,
        label: Optional[str] = None,
        label_code: Optional[Dict[str, int]] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
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
        self.slice_index = slice_index
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
        self.preprocessing.discarded_slices = compute_discarded_slices(
            preprocessing_dict["discarded_slices"]
        )

    @property
    def elem_index(self):
        return self.slice_index

    def __getitem__(self, idx):
        participant, session, cohort, slice_idx, label, domain = self._get_meta_data(
            idx
        )
        slice_idx = slice_idx + self.preprocessing.discarded_slices[0]
        image_path = self._get_image_path(participant, session, cohort)

        if self.preprocessing.save_features:
            slice_dir = image_path.parent.as_posix().replace(
                "image_based", f"{self.preprocessing.extract_method.value}_based"
            )
            slice_filename = extract_slice_path(
                image_path,
                self.preprocessing.slice_direction,
                self.preprocessing.slice_mode,
                slice_idx,
            )
            slice_tensor = torch.load(Path(slice_dir) / slice_filename)

        else:
            image_path = self._get_image_path(participant, session, cohort)
            image = torch.load(image_path)
            slice_tensor = extract_slice_tensor(
                image,
                self.preprocessing.slice_direction,
                self.preprocessing.slice_mode,
                slice_idx,
            )

        if self.data.transformations:
            slice_tensor = self.data.transformations(slice_tensor)

        if self.data.augmentation_transformations and not self.eval_mode:
            slice_tensor = self.data.augmentation_transformations(slice_tensor)

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

        if self.preprocessing.num_slices is not None:
            return self.preprocessing.num_slices

        image = self._get_full_image()
        return (
            image.size(int(self.preprocessing.slice_direction) + 1)
            - self.preprocessing.discarded_slices[0]
            - self.preprocessing.discarded_slices[1]
        )


def return_dataset(
    input_dir: Path,
    data_df: pd.DataFrame,
    preprocessing_dict: Dict[str, Any],
    all_transformations: Optional[Callable],
    label: Optional[str] = None,
    label_code: Optional[Dict[str, int]] = None,
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
        raise NotImplementedError(
            f"Mode {preprocessing_dict['mode']} is not implemented."
        )


################################
# TSV files loaders
################################
def load_data_test(test_path: Path, diagnoses_list, baseline=True, multi_cohort=False):
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
        if not test_path.suffix == ".tsv":
            raise ClinicaDLArgumentError(
                "If multi_cohort is given, the TSV_DIRECTORY argument should be a path to a TSV file."
            )
        else:
            tsv_df = pd.read_csv(test_path, sep="\t")
            check_multi_cohort_tsv(tsv_df, "labels")
            test_df = pd.DataFrame()
            found_diagnoses = set()
            for idx in range(len(tsv_df)):
                cohort_name = tsv_df.loc[idx, "cohort"]
                cohort_path = Path(tsv_df.loc[idx, "path"])
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
        if test_path.suffix == ".tsv":
            tsv_df = pd.read_csv(test_path, sep="\t")
            multi_col = {"cohort", "path"}
            if multi_col.issubset(tsv_df.columns.values):
                raise ClinicaDLConfigurationError(
                    "To use multi-cohort framework, please add 'multi_cohort=true' in your configuration file or '--multi_cohort' flag to the command line."
                )
        test_df = load_data_test_single(test_path, diagnoses_list, baseline=baseline)
        test_df["cohort"] = "single"

    return test_df


def load_data_test_single(test_path: Path, diagnoses_list, baseline=True):
    if test_path.suffix == ".tsv":
        test_df = pd.read_csv(test_path, sep="\t")
        if "diagnosis" not in test_df.columns.values:
            raise ClinicaDLTSVError(
                f"'diagnosis' column must be present in TSV file {test_path}."
            )
        test_df = test_df[test_df.diagnosis.isin(diagnoses_list)]
        if len(test_df) == 0:
            raise ClinicaDLTSVError(
                f"Diagnoses wanted {diagnoses_list} were not found in TSV file {test_path}."
            )
        return test_df

    test_df = pd.DataFrame()

    if baseline:
        if not (test_path.parent / "train_baseline.tsv").is_file():
            if not (test_path.parent / "labels_baseline.tsv").is_file():
                raise ClinicaDLTSVError(
                    f"There is no train_baseline.tsv nor labels_baseline.tsv in your folder {test_path.parents[0]} "
                )
            else:
                test_path = test_path.parent / "labels_baseline.tsv"
        else:
            test_path = test_path.parent / "train_baseline.tsv"
    else:
        if not (test_path.parent / "train.tsv").is_file():
            if not (test_path.parent / "labels.tsv").is_file():
                raise ClinicaDLTSVError(
                    f"There is no train.tsv or labels.tsv in your folder {test_path.parent} "
                )
            else:
                test_path = test_path.parent / "labels.tsv"
        else:
            test_path = test_path.parent / "train.tsv"

    test_df = pd.read_csv(test_path, sep="\t")
    test_df = test_df[test_df.diagnosis.isin(diagnoses_list)]
    test_df.reset_index(inplace=True, drop=True)

    return test_df


def check_multi_cohort_tsv(tsv_df: pd.DataFrame, purpose: str) -> None:
    """
    Checks that a multi-cohort TSV file is valid.

    Args:
        tsv_df (pd.DataFrame): DataFrame of multi-cohort definition.
        purpose (str): what the TSV file describes (CAPS or TSV).
    Raises:
        ValueError: if the TSV file is badly formatted.
    """
    mandatory_col = ("cohort", "diagnoses", "path")
    if purpose.upper() == "CAPS":
        mandatory_col = ("cohort", "path")
    if not set(mandatory_col).issubset(tsv_df.columns.values):
        raise ClinicaDLTSVError(
            f"Columns of the TSV file used for {purpose} location must include {mandatory_col}"
        )
