# coding: utf8

import abc
from os import path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from clinicadl.utils.inputs import FILENAME_TYPE, MASK_PATTERN

#################################
# Datasets loaders
#################################


class CapsDataset(Dataset):
    """Abstract class for all derived CapsDatasets."""

    def __init__(
        self,
        caps_directory,
        data_df,
        preprocessing,
        transformations,
        label_presence,
        label=None,
        label_code=None,
        augmentation_transformations=None,
        multi_cohort=False,
    ):
        self.caps_dict = self.create_caps_dict(caps_directory, multi_cohort)
        self.transformations = transformations
        self.augmentation_transformations = augmentation_transformations
        self.eval_mode = False
        self.label_presence = label_presence
        self.label = label
        self.label_code = label_code
        self.preprocessing = preprocessing

        if not hasattr(self, "elem_index"):
            raise ValueError(
                "Child class of CapsDataset must set elem_index attribute."
            )
        if not hasattr(self, "mode"):
            raise ValueError("Child class of CapsDataset must set mode attribute.")
        # Check the format of the tsv file here
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

    def label_fn(self, target):
        """
        Returns the label value usable in criterion.

        Args:
            target (str or float or int): value of the target.
        Returns:
            label (int or float): value of the label usable in criterion.
        """
        # Reconstruction case (no label)
        if self.label is None:
            return None
        # Regression case (no label code)
        elif self.label_code is None:
            return np.float32([target])
        # Classification case (label + label_code dict)
        else:
            return self.label_code[target]

    def __len__(self):
        return len(self.df) * self.elem_per_image

    @staticmethod
    def create_caps_dict(caps_directory, multi_cohort):

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

    def _get_path(self, participant, session, cohort, mode="image"):
        """
        Gets the path to the tensor image (*.pt)

        Args:
            participant (str): ID of the participant.
            session (str): ID of the session.
            cohort (str): Name of the cohort.
            mode (str): Type of mode used (image, patch, slice or roi).
        Returns:
            image_path (str): path to the image
        """

        if cohort not in self.caps_dict.keys():
            raise ValueError(
                "Cohort names in labels and CAPS definitions do not match."
            )

        if self.preprocessing == "t1-linear":
            image_path = path.join(
                self.caps_dict[cohort],
                "subjects",
                participant,
                session,
                "deeplearning_prepare_data",
                "%s_based" % mode,
                "t1_linear",
                participant + "_" + session + FILENAME_TYPE["cropped"] + ".pt",
            )
        elif self.preprocessing == "t1-linear-downsampled":
            image_path = path.join(
                self.caps_dict[cohort],
                "subjects",
                participant,
                session,
                "deeplearning_prepare_data",
                "%s_based" % mode,
                "t1_linear",
                participant + "_" + session + FILENAME_TYPE["downsampled"] + ".pt",
            )
        elif self.preprocessing == "t1-extensive":
            image_path = path.join(
                self.caps_dict[cohort],
                "subjects",
                participant,
                session,
                "deeplearning_prepare_data",
                "%s_based" % mode,
                "t1_extensive",
                participant + "_" + session + FILENAME_TYPE["skull_stripped"] + ".pt",
            )
        elif self.preprocessing == "t1-volume":
            image_path = path.join(
                self.caps_dict[cohort],
                "subjects",
                participant,
                session,
                "deeplearning_prepare_data",
                "%s_based" % mode,
                "custom",
                participant + "_" + session + FILENAME_TYPE["gm_maps"] + ".pt",
            )
        elif self.preprocessing == "shepplogan":
            image_path = path.join(
                self.caps_dict[cohort],
                "subjects",
                "%s_%s%s.pt" % (participant, session, FILENAME_TYPE["shepplogan"]),
            )
        else:
            raise NotImplementedError(
                "The path to preprocessing %s is not implemented" % self.preprocessing
            )

        return image_path

    def _get_meta_data(self, idx):
        """
        Gets all meta data necessary to compute the path with _get_path

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

    def _get_full_image(self):
        """
        Allows to get the an example of the image mode corresponding to the dataset.
        Useful to compute the number of elements if mode != image.

        Returns:
            image (torch.Tensor) tensor of the full image.
        """
        import nibabel as nib

        from clinicadl.generate.generate_utils import find_image_path as get_nii_path

        participant_id = self.df.loc[0, "participant_id"]
        session_id = self.df.loc[0, "session_id"]
        cohort = self.df.loc[0, "cohort"]

        try:
            image_path = self._get_path(
                participant_id, session_id, cohort, mode="image"
            )
            image = torch.load(image_path)
        except FileNotFoundError:
            image_path = get_nii_path(
                self.caps_dict,
                participant_id,
                session_id,
                cohort=cohort,
                preprocessing=self.preprocessing,
            )
            image_nii = nib.load(image_path)
            image_np = image_nii.get_fdata()
            image = ToTensor()(image_np)

        return image

    @abc.abstractmethod
    def __getitem__(self, idx):
        """
        Gets the sample containing all the information needed for training and testing tasks.

        Args:
            idx (int): row number of the meta-data contained in self.df
        Returns:
            Dict[str, Any]: dictionary with following items:
                - "image" (torch.Tensor): the input given to the model,
                - "label" (int or float): the label used in criterion,
                - "participant_id" (str): ID of the participant,
                - "session_id" (str): ID of the session,
                - f"{self.mode}_id" (int): number of the element,
                - "image_path": path to the image loaded in CAPS.

        """
        pass

    @abc.abstractmethod
    def num_elem_per_image(self):
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
        caps_directory,
        data_file,
        preprocessing="t1-linear",
        train_transformations=None,
        label_presence=True,
        label=None,
        label_code=None,
        all_transformations=None,
        multi_cohort=False,
    ):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            train_transformations (callable, optional): Optional transform to be applied only on training mode.
            label_presence (bool): If True the diagnosis will be extracted from the given DataFrame.
            label (str): Name of the column in data_df containing the label.
            label_code (Dict[str, int]): label code that links the output node number to label value.
            all_transformations (callable, options): Optional transform to be applied during training and evaluation.
            multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.

        """
        self.mode = "image"
        super().__init__(
            caps_directory,
            data_file,
            preprocessing,
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

        image_path = self._get_path(participant, session, cohort, "image")
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
        caps_directory,
        data_file,
        patch_size,
        stride_size,
        train_transformations=None,
        prepare_dl=False,
        patch_index=None,
        preprocessing="t1-linear",
        label_presence=True,
        label=None,
        label_code=None,
        all_transformations=None,
        multi_cohort=False,
    ):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            train_transformations (callable, optional): Optional transform to be applied only on training mode.
            prepare_dl (bool): If true pre-extracted patches will be loaded.
            patch_index (int, optional): If a value is given the same patch location will be extracted for each image.
                else the dataset will load all the patches possible for one image.
            patch_size (int): size of the regular cubic patch.
            stride_size (int): length between the centers of two patches.
            label_presence (bool): If True the diagnosis will be extracted from the given DataFrame.
            label (str): Name of the column in data_df containing the label.
            label_code (Dict[str, int]): label code that links the output node number to label value.
            all_transformations (callable, options): Optional transform to be applied during training and evaluation.
            multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.

        """
        if preprocessing == "shepplogan":
            raise ValueError(
                "Patch mode is not available for preprocessing %s" % preprocessing
            )
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.patch_index = patch_index
        self.mode = "patch"
        self.prepare_dl = prepare_dl
        super().__init__(
            caps_directory,
            data_file,
            preprocessing,
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

        if self.prepare_dl:
            patch_path = path.join(
                self._get_path(participant, session, cohort, "patch")[0:-7]
                + "_patchsize-"
                + str(self.patch_size)
                + "_stride-"
                + str(self.stride_size)
                + "_patch-"
                + str(patch_idx)
                + "_T1w.pt"
            )

            image = torch.load(patch_path)
        else:
            image_path = self._get_path(participant, session, cohort, "image")
            full_image = torch.load(image_path)
            image = self.extract_patch_from_mri(full_image, patch_idx)

        if self.transformations:
            image = self.transformations(image)

        if self.augmentation_transformations and not self.eval_mode:
            image = self.augmentation_transformations(image)

        sample = {
            "image": image,
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

    def extract_patch_from_mri(self, image_tensor, patch_idx):
        """
        Extracts the patch corresponding to patch_idx

        Args:
            image_tensor (torch.Tensor): tensor of the full image.
            patch_idx (int): Index of the patch wanted.
        Returns:
            extracted_patch (torch.Tensor): the tensor of the patch.
        """

        patches_tensor = (
            image_tensor.unfold(1, self.patch_size, self.stride_size)
            .unfold(2, self.patch_size, self.stride_size)
            .unfold(3, self.patch_size, self.stride_size)
            .contiguous()
        )
        patches_tensor = patches_tensor.view(
            -1, self.patch_size, self.patch_size, self.patch_size
        )
        extracted_patch = patches_tensor[patch_idx, ...].unsqueeze_(0).clone()

        return extracted_patch


class CapsDatasetRoi(CapsDataset):
    def __init__(
        self,
        caps_directory,
        data_file,
        roi_list=None,
        cropped_roi=True,
        roi_index=None,
        preprocessing="t1-linear",
        train_transformations=None,
        prepare_dl=False,
        label_presence=True,
        label=None,
        label_code=None,
        all_transformations=None,
        multi_cohort=False,
    ):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            roi_list (list): Defines the regions used in the classification.
            cropped_roi (bool): If True the image is cropped according to the smallest bounding box possible.
            roi_index (int, optional): If a value is given the same region will be extracted for each image.
                else the dataset will load all the regions possible for one image.
            preprocessing (string): Defines the path to the data in CAPS.
            train_transformations (callable, optional): Optional transform to be applied only on training mode.
            prepare_dl (bool): If true pre-extracted patches will be loaded.
            label_presence (bool): If True the diagnosis will be extracted from the given DataFrame.
            label (str): Name of the column in data_df containing the label.
            label_code (Dict[str, int]): label code that links the output node number to label value.
            all_transformations (callable, options): Optional transform to be applied during training and evaluation.
            multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.

        """
        if preprocessing == "shepplogan":
            raise ValueError(
                "ROI mode is not available for preprocessing %s" % preprocessing
            )
        self.roi_index = roi_index
        self.mode = "roi"
        self.roi_list = roi_list
        self.cropped_roi = cropped_roi
        self.prepare_dl = prepare_dl
        self.mask_list = self.find_masks(caps_directory, preprocessing)
        super().__init__(
            caps_directory,
            data_file,
            preprocessing,
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

        if self.prepare_dl:
            if self.roi_list is None:
                raise NotImplementedError(
                    "The extraction of ROIs prior to training is not implemented for default ROIs."
                    "Please disable --use_extracted_rois or precise the regions in --roi_names."
                )

            # read the regions directly
            roi_path = self._get_path(participant, session, cohort, "roi")
            roi_path = self.compute_roi_filename(roi_path, roi_idx)
            patch = torch.load(roi_path)

        else:
            image_path = self._get_path(participant, session, cohort, "image")
            image = torch.load(image_path)
            patch = self.extract_roi_from_mri(image, roi_idx)

        if self.transformations:
            patch = self.transformations(patch)

        if self.augmentation_transformations and not self.eval_mode:
            patch = self.augmentation_transformations(patch)

        sample = {
            "image": patch,
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

    def extract_roi_from_mri(self, image_tensor, roi_idx):
        """
        Extracts the region of interest corresponding to the roi_idx-th mask given to the dataset

        Args:
            image_tensor (torch.Tensor): tensor of the full image.
            roi_idx (int): Index of the region wanted.
        Returns:
            extracted_roi (torch.Tensor): the tensor of the region.
        """

        if self.roi_list is None:

            if self.preprocessing == "t1-linear":
                if roi_idx == 1:
                    # the center of the left hippocampus
                    crop_center = (61, 96, 68)
                else:
                    # the center of the right hippocampus
                    crop_center = (109, 96, 68)
            else:
                raise NotImplementedError(
                    "The extraction of hippocampi was not implemented for "
                    "preprocessing %s" % self.preprocessing
                )
            crop_size = (50, 50, 50)  # the output cropped hippocampus size

            if self.cropped_roi:

                extracted_roi = image_tensor[
                    :,
                    crop_center[0]
                    - crop_size[0] // 2 : crop_center[0]
                    + crop_size[0] // 2 :,
                    crop_center[1]
                    - crop_size[1] // 2 : crop_center[1]
                    + crop_size[1] // 2 :,
                    crop_center[2]
                    - crop_size[2] // 2 : crop_center[2]
                    + crop_size[2] // 2 :,
                ].clone()

            else:
                raise NotImplementedError(
                    "The uncropped option for the default ROI was not implemented."
                )

        else:
            roi_mask = self.mask_list[roi_idx]
            extracted_roi = image_tensor * roi_mask
            if self.cropped_roi:
                extracted_roi = extracted_roi[
                    np.ix_(
                        roi_mask.any((1, 2, 3)),
                        roi_mask.any((0, 2, 3)),
                        roi_mask.any((0, 1, 3)),
                        roi_mask.any((0, 1, 2)),
                    )
                ]

        return extracted_roi.float()

    def find_masks(self, caps_directory, preprocessing):
        """Loads the masks necessary to regions extraction"""
        import nibabel as nib

        # TODO should be mutualized with deeplearning-prepare-data
        templates_dict = {
            "t1-linear": "MNI152NLin2009cSym",
            "t1-volume": "Ixi549Space",
            "t1-extensive": "Ixi549Space",
        }

        if self.prepare_dl or self.roi_list is None:
            return None
        else:
            mask_list = []
            for roi in self.roi_list:
                template = templates_dict[preprocessing]
                if preprocessing == "t1-linear":
                    mask_pattern = MASK_PATTERN["cropped"]
                elif preprocessing == "t1-volume":
                    mask_pattern = MASK_PATTERN["gm_maps"]
                elif preprocessing == "t1-extensive":
                    mask_pattern = MASK_PATTERN["skull_stripped"]
                else:
                    raise NotImplementedError(
                        "Roi extraction for %s preprocessing was not implemented."
                        % preprocessing
                    )

                mask_path = path.join(
                    caps_directory,
                    "masks",
                    "tpl-%s" % template,
                    "tpl-%s%s_roi-%s_mask.nii.gz" % (template, mask_pattern, roi),
                )
                mask_nii = nib.load(mask_path)
                mask_list.append(mask_nii.get_fdata())

        return mask_list

    def compute_roi_filename(self, image_path, roi_index):
        # TODO should be mutualized with deeplearning-prepare-data
        from os import path

        image_dir = path.dirname(image_path)
        image_filename = path.basename(image_path)
        image_descriptors = image_filename.split("_")
        if "desc-Crop" not in image_descriptors and self.cropped_roi:
            image_descriptors = self.insert_descriptor(
                image_descriptors, "desc-CropRoi", "space"
            )

        elif "desc-Crop" in image_descriptors:
            image_descriptors = [
                descriptor
                for descriptor in image_descriptors
                if descriptor != "desc-Crop"
            ]
            if self.cropped_roi:
                image_descriptors = self.insert_descriptor(
                    image_descriptors, "desc-CropRoi", "space"
                )
            else:
                image_descriptors = self.insert_descriptor(
                    image_descriptors, "desc-CropImage", "space"
                )

        return (
            path.join(image_dir, "_".join(image_descriptors))[0:-7]
            + f"_roi-{self.roi_list[roi_index]}_T1w.pt"
        )

    @staticmethod
    def insert_descriptor(image_descriptors, descriptor_to_add, key_to_follow):
        # TODO should be mutualized with deeplearning-prepare-data

        for i, desc in enumerate(image_descriptors):
            if key_to_follow in desc:
                image_descriptors.insert(i + 1, descriptor_to_add)

        return image_descriptors


class CapsDatasetSlice(CapsDataset):
    def __init__(
        self,
        caps_directory,
        data_file,
        slice_index=None,
        preprocessing="t1-linear",
        train_transformations=None,
        mri_plane=0,
        prepare_dl=False,
        discarded_slices=20,
        label_presence=True,
        label=None,
        label_code=None,
        all_transformations=None,
        multi_cohort=False,
    ):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            slice_index (int, optional): If a value is given the same slice will be extracted for each image.
                else the dataset will load all the slices possible for one image.
            train_transformations (callable, optional): Optional transform to be applied only on training mode.
            prepare_dl (bool): If true pre-extracted patches will be loaded.
            mri_plane (int): Defines which mri plane is used for slice extraction.
            discarded_slices (int or list): number of slices discarded at the beginning and the end of the image.
                If one single value is given, the same amount is discarded at the beginning and at the end.
            label_presence (bool): If True the diagnosis will be extracted from the given DataFrame.
            label (str): Name of the column in data_df containing the label.
            label_code (Dict[str, int]): label code that links the output node number to label value.
            all_transformations (callable, options): Optional transform to be applied during training and evaluation.
            multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.
        """
        # Rename MRI plane
        if preprocessing == "shepplogan":
            raise ValueError(
                "Slice mode is not available for preprocessing %s" % preprocessing
            )
        self.slice_index = slice_index
        self.mri_plane = mri_plane
        self.direction_list = ["sag", "cor", "axi"]
        if self.mri_plane >= len(self.direction_list):
            raise ValueError(
                "mri_plane value %i > %i" % (self.mri_plane, len(self.direction_list))
            )

        # Manage discarded_slices
        if isinstance(discarded_slices, int):
            discarded_slices = [discarded_slices, discarded_slices]
        if isinstance(discarded_slices, list) and len(discarded_slices) == 1:
            discarded_slices = discarded_slices * 2
        self.discarded_slices = discarded_slices

        self.mode = "slice"
        self.prepare_dl = prepare_dl
        super().__init__(
            caps_directory,
            data_file,
            preprocessing,
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

        if self.prepare_dl:
            # read the slices directly
            slice_path = path.join(
                self._get_path(participant, session, cohort, "slice")[0:-7]
                + "_axis-%s" % self.direction_list[self.mri_plane]
                + "_channel-rgb_slice-%i_T1w.pt" % slice_idx
            )
            image = torch.load(slice_path)
        else:
            image_path = self._get_path(participant, session, cohort, "image")
            full_image = torch.load(image_path)
            image = self.extract_slice_from_mri(full_image, slice_idx)

        if self.transformations:
            image = self.transformations(image)

        if self.augmentation_transformations and not self.eval_mode:
            image = self.augmentation_transformations(image)

        sample = {
            "image": image,
            "label": label,
            "participant_id": participant,
            "session_id": session,
            "slice_id": slice_idx,
        }

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1

        image = self._get_full_image()
        return (
            image.size(self.mri_plane + 1)
            - self.discarded_slices[0]
            - self.discarded_slices[1]
        )

    def extract_slice_from_mri(self, image, index_slice):
        """
        This function extracts one slice along one axis and creates a RGB image
        (the slice is duplicated in each channel).

        To note:
        Axial_view = "[:, :, slice_i]"
        Coronal_view = "[:, slice_i, :]"
        Sagittal_view= "[slice_i, :, :]"

        Args:
            image (torch.Tensor): tensor of the full image.
            index_slice (int): index of the wanted slice.
        Returns:
            triple_slice (torch.Tensor): tensor of the slice with 3 channels.
        """
        image = image.squeeze(0)
        simple_slice = image[(slice(None),) * self.mri_plane + (index_slice,)]
        triple_slice = torch.stack((simple_slice, simple_slice, simple_slice))

        return triple_slice


def return_dataset(
    mode,
    input_dir,
    data_df,
    preprocessing,
    all_transformations,
    params,
    label=None,
    label_code=None,
    train_transformations=None,
    cnn_index=None,
    label_presence=True,
    multi_cohort=False,
    prepare_dl=False,
):
    """
    Return appropriate Dataset according to given options.
    Args:
        mode (str): input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
        input_dir (str): path to a directory containing a CAPS structure.
        data_df (pd.DataFrame): List subjects, sessions and diagnoses.
        preprocessing (str): type of preprocessing wanted ('t1-linear' or 't1-extensive')
        train_transformations (callable, optional): Optional transform to be applied during training only.
        all_transformations (callable, optional): Optional transform to be applied during training and evaluation.
        params (clinicadl.MapsManager): options used by specific modes.
        label (str): Name of the column in data_df containing the label.
        label_code (Dict[str, int]): label code that links the output node number to label value.
        cnn_index (int): Index of the CNN in a multi-CNN paradigm (optional).
        label_presence (bool): If True the diagnosis will be extracted from the given DataFrame.
        multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.
        prepare_dl (bool): If true pre-extracted slices / patches / regions will be loaded.

    Returns:
         (Dataset) the corresponding dataset.
    """

    if cnn_index is not None and mode in ["image"]:
        raise ValueError("Multi-CNN is not implemented for %s mode." % mode)

    if mode == "image":
        return CapsDatasetImage(
            input_dir,
            data_df,
            preprocessing,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    elif mode == "patch":
        return CapsDatasetPatch(
            input_dir,
            data_df,
            params.patch_size,
            params.stride_size,
            preprocessing=preprocessing,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            prepare_dl=prepare_dl,
            patch_index=cnn_index,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    elif mode == "roi":
        return CapsDatasetRoi(
            input_dir,
            data_df,
            roi_list=params.roi_list,
            cropped_roi=not params.uncropped_roi,
            preprocessing=preprocessing,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            prepare_dl=prepare_dl,
            roi_index=cnn_index,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    elif mode == "slice":
        return CapsDatasetSlice(
            input_dir,
            data_df,
            preprocessing=preprocessing,
            train_transformations=train_transformations,
            all_transformations=all_transformations,
            mri_plane=params.slice_direction,
            prepare_dl=prepare_dl,
            discarded_slices=params.discarded_slices,
            slice_index=cnn_index,
            label_presence=label_presence,
            label=label,
            label_code=label_code,
            multi_cohort=multi_cohort,
        )
    else:
        raise ValueError("Mode %s is not implemented." % mode)


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


def get_transforms(mode, minmaxnormalization=True, data_augmentation=None):
    """
    Outputs the transformations that will be applied to the dataset

    Args:
        mode (str): input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
        minmaxnormalization (bool): if True will perform MinMaxNormalization.
        data_augmentation (List[str]): list of data augmentation performed on the training set.
    Returns:
    - container transforms.Compose including transforms to apply in train and evaluation mode.
    - container transforms.Compose including transforms to apply in evaluation mode only.
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

    if mode == "slice":
        trg_size = (224, 224)
        transformations_list += [
            transforms.ToPILImage(),
            transforms.Resize(trg_size),
            transforms.ToTensor(),
        ]

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
