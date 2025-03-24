# coding: utf8
from __future__ import annotations

from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torchio as tio
from torch.utils.data import Dataset

from clinicadl.dictionary.suffixes import PT
from clinicadl.dictionary.words import (
    AFFINE,
    FIRST_INDEX,
    IMAGE,
    LABEL,
    LAST_INDEX,
    N_SAMPLES,
    PARTICIPANT,
    PARTICIPANT_ID,
    SESSION,
    SESSION_ID,
)
from clinicadl.transforms.extraction import Sample
from clinicadl.transforms.transforms import Transforms
from clinicadl.tsvtools.utils import (
    check_df,
    tsv_to_df,
)
from clinicadl.utils.enum import ExtractionMethod
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
    ClinicaDLTSVError,
)
from clinicadl.utils.typing import DataType, PathType

from ..datatype.preprocessing import Preprocessing, T1Linear
from ..readers.caps_reader import CapsReader
from ..structures import Column, DataPoint, Mask
from ..tensor_conversion import TensorConversion

logger = getLogger("clinicadl.caps_dataset")


class CapsDataset(Dataset):
    """
    CapsDataset is a custom PyTorch Dataset class for working with neuroimaging data in CAPS format.\n

    The user specifies the type of data he wants to work on via `preprocessing`, the (participant, session)
    pairs he wants to work on via `data`, the transforms he wants to apply on images via `transforms` and
    where to find the label (scalars or masks) associated to the images via `label`. Some transforms may
    also need masks (e.g. setting background to 0 outside a mask), which can be specified via `masks`.\n

    A CapsDataset works with tensors, so, before manipulating data, NIfTI files must be converted to PyTorch's
    `.pt` format with to `to_tensors` method. If conversion was already performed, `read_tensor_conversion`
    must be called.

    Parameters
    ----------
    caps_directory : PathType
        Path to the CAPS directory containing the neuroimaging data. A string or a Path object.
    preprocessing : Preprocessing, (optional, default=PreprocessingT1())
        Description of the preprocessing steps applied to the data. Default is Clinica's `t1-linear`
        pipeline. See :py:class:`clinicadl.data.datatype.Preprocessing`.
    data : Optional[DataType], (optional, default=None)
        A DataFrame (or a path to a TSV file containing the dataframe) with the list of participant/session
        pairs to consider, as well as any other relevant information (e.g. the labels for classification or
        regression).\n
        Only participant/session pairs in this TSV file will be in the CapsDataset.\n
        If `None`, all participant/session pairs in `caps_directory` will be used. Besides, a TSV file
        named will be created in `caps_directory`, with the list of all participant/session
        pairs in the directory. The name of the created TSV depends on the preprocessing, but it will
        always start with `overview` (e.g. `overview_t1-linear_cropped.tsv`,
        `overview_pet-linear_18FFDG_pons2.tsv`, etc.).
    .. warning::
        Beware that your TSV files inside `caps_directory` may be overwritten. A good practice is not
        to name your own TSV files with a name starting with `overview`.
    label : Optional[str], (optional, default=None)
        A potential label related to the image.\n
        If `label` and `data` are not `None`, CapsDataset will look for a column with that name in `data`.
        It expects to find the associated column, with floats (regression) or integers (classification).\n
        If there is no such column in `data` (or if `data` is `None`), CapsDataset will look for
        masks with that label as a suffix. The label is thus a mask (segmentation). For example, if
        the image of the participant `sub-001` for the session `ses-M000` is in
        `sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz` and `label="seg"`, it will look for the associated
        mask in `sub-001/ses-M000/sub-001_ses-M000_seg.nii.gz`.\n
        If `None`, no label will be used (e.g. reconstruction).
    transforms : Transforms, (optional, default=Transforms())
        Transformation pipeline to apply to the data during loading. Default will only apply `NaN` removal
        to images. See :py:class:`clinicadl.transforms.Transforms`.
    masks : Optional[list[PathType]], (optional, default=None)
        Potential masks that are useful to compute some transforms.
        A mask can be either a suffix (image-specific masks) or a file in the `masks` folder of
        `caps_directory` (common masks).\n
        For example, if `masks=["brain", "hippocampus.nii.gz"]`:
        - For the mask `"hippocampus.nii.gz"`, a file is passed. Therefore, it is understood as a mask common
        to all images. So, CapsDataset will simply get the mask in `{caps_directory}/masks/hippocampus.nii.gz`.\n
        - For `"brain"`, a suffix is passed. Therefore, it is understood as an image-specific mask:
        if the image is in `sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz`, it will look for the masks in
        `sub-001/ses-M000/sub-001_ses-M000_brain.nii.gz`.\n
        To use the masks in transforms, mention "brain" and "hippocampus.nii.gz" (see examples).

    Raises
    ------
    ClinicaDLArgumentError
        if `caps_directory` if not a directory.
    ClinicaDLArgumentError
        If `data` is not a DataFrame, a path or `None`.
    ClinicaDLTSVError
        If `data` is a TSV file that does not exist.
    ClinicaDLTSVError
        If the DataFrame in `data` is empty.
    ClinicaDLTSVError
        If the DataFrame in `data` does not contain the columns `"participant_id"`
        and `"session_id"`.
    ClinicaDLTSVError
        If the DataFrame in `data` contains duplicated (participant_id, session_id) pairs.
    ClinicaDLConfigurationError
        If the data does not match the preprocessing configuration.
    ClinicaDLArgumentError
        If `label` is not a string or `None`.
    FileNotFoundError
        If `masks` contain paths that do not match any files.

    Examples
    --------
    >>> # data are as follows:
    >>> # mycaps
    >>> # ├── masks
    >>> # │   └── leftHippocampus.nii.gz
    >>> # ├── pet_data.tsv
    >>> # └── subjects
    >>> #     ├── sub-000
    >>> #     │   └── ses-M000
    >>> #     │       └── pet_linear
    >>> #     │           ├── sub-000_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_brain.nii.gz
    >>> #     │           ├── sub-000_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz
    >>> #     │           └── sub-000_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_seg.nii.gz
    >>> #         ...
    >>> #     ...
    >>> from clinicadl.data import CapsDataset
    >>> from clinicadl.data.datatype import PETLinear
    >>> from clinicadl.transforms import Transforms, get_transform_config
    >>> from clinicadl.transforms.extraction import Patch
    >>> normalization = get_transform_config("ZNormalization", masking_method="brain")
    >>> mask = get_transform_config("Mask", masking_method="leftHippocampus")
    >>> flip = get_transform_config("RandomFlip", flip_probability=0.3)
    >>> dataset = CapsDataset(
            caps_directory="mycaps",
            preprocessing=PETLinear(
                tracer="18FAV45", use_uncropped_image=True, suvr_reference_region="pons2"
            ),
            data="mycaps/pet_data.tsv",
            transforms=Transforms(
                extraction=Patch(patch_size=32, stride=32),
                image_transforms=[normalization, mask],
                sample_transforms=[],
                augmentations=[flip],
            ),
            label="seg",    # labels are here images (in files "sub-*_ses-*_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_seg.nii.gz")
            masks=["brain", "leftHippocampus.nii.gz"],  # define masks used in transforms
        )                                               # 'brain' is image-specific (in files "sub-*_ses-*_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_brain.nii.gz")
                                                        # 'leftHippocampus.nii.gz' is a common mask (in "masks/leftHippocampus.nii.gz")
    >>> dataset.to_tensors("pet_masked")
    """

    def __init__(
        self,
        caps_directory: PathType,
        preprocessing: Preprocessing = T1Linear(),
        data: Optional[DataType] = None,
        label: Optional[str] = None,
        transforms: Transforms = Transforms(),
        masks: Optional[list[PathType]] = None,
    ):
        self.eval_mode = False
        self.caps_reader = CapsReader(caps_directory)
        self.directory = Path(caps_directory)
        self.preprocessing = preprocessing
        self.transforms = transforms
        (
            self.image_transform,
            self.sample_transform,
            self.augmentation,
        ) = transforms.get_transforms()
        self.extraction = transforms.extraction
        self.df = self._get_df_from_input(data)
        self.caps_reader.check_preprocessing(
            self.get_participant_session_couples(), self.preprocessing
        )
        self.label = self._check_label(label)
        self.individual_masks, self.common_masks = self._read_masks(masks)
        self.tensor_conversion: TensorConversion = TensorConversion(self)

        self.common_masks_tensors: list[Mask] = []

    def read_tensor_conversion(
        self, json_name: str, check_transforms: bool = True
    ) -> None:
        """
        To read an old tensor conversion. The function will check that
        the old conversion works with the current CapsDataset, i.e. that
        the images of the current (participant, session) pairs have been converted
        to tensors, as well as masks.
        If transformed images have been saved, it will also check that the transforms
        applied before conversion match the image transforms of the current CapsDataset,
        unless `check_transforms` is False.\n

        See :py:func:`clinicadl.data.CapsDataset.to_tensors` for more information on
        conversion to tensors.

        Parameters
        ----------
        json_name : str
            the name of the json file (without `.json` suffix) in the folder 'tensor_extraction'
            of the caps directory describing the tensor conversion.
        check_transforms : bool (optional, default=True)
            whether to checks if the image transforms potentially applied before tensor conversion
            match the current ones. Useful when you use custom transforms (i.e. transforms
            not in ClinicaDL), which cannot read by ClinicaDL and thus cannot be checked.\n
            ..note::If 'convert_to_tensors' was run with `save_transforms=False`, no check will
            be performed as the tensors saved have not been transformed.
            ..warning::To use carefully: you need to be sure that the transforms match.

        Raises
        ------
        FileNotFoundError
            if there is no json file named `json_name` in the `tensor_extraction` folder.
        ClinicaDLTensorConversionError
            if the conversion mentioned in the json file doesn't work with the
            current CapsDataset (not the same preprocessing, images not all converted, transforms
            mismatch, etc.).
        """
        self.tensor_conversion.read_conversion(json_name, check_transforms)
        self._load_pt_masks()
        self._count_samples()

    def to_tensors(
        self,
        json_name: PathType = "tensor_conversion",
        save_transforms: bool = True,
        n_proc: int = 1,
        ignore_spacing: bool = False,
        raise_warnings: bool = True,
    ) -> None:
        """
        Converts NIfTI files to tensors (in PyTorch's `.pt` format), the only format that a
        CapsDataset can manipulate.
        This is a mandatory step before manipulating a CapsDataset, as some checks on data will
        be performed before conversion (shape consistency, voxel spacing consistency, etc.).
        Conversion to tensors also significantly speeds up data loading during training or
        inference.\n

        The user have the possibility to store transformed images, i.e. images on which
        image transforms have already been applied (see: :py:class:`clinicadl.transforms.Transforms`).
        This practice will speed up dataloading during training or inference as the images don't have
        to be transformed each time they are loaded. The drawback is that the saved images can't be
        used by a CapsDataset with other image transforms.

        Parameters
        ----------
        json_name : str (optional, default="tensor_conversion")
            the name of the json file where the information on the conversion
            (e.g. transforms applied) will be stored. The full path of
            the json file will be `{caps_directory}/prepare_data/tensor_conversion/{json_name}.json`.\n
            If the file already exists, ClinicaDL will try to merge the old
            tensor conversion with the new one, if they concern the same type of data (same
            preprocessing, same transforms applied, etc.), otherwise an error will be raised.
        save_transforms : bool (optional, default=True)
            whether to save raw images as tensors (False) or images on which were applied image
            transforms (True). Saving transformed images will speed up dataloading. However transformed
            images are specific to a set of transforms, so they cannot be used by any future CapsDataset.
        n_proc : int (optional, default=1)
            number of cores to use to parallelize the conversion.
        ignore_spacing : bool (optional, default=False)
            whether to ignore the check made on voxel spacings. If False, it will make sure that all
            images have the same voxel spacing before converting them.
            ..warning::In most medical image applications, all the images should have the same
            voxel spacing. Be sure that you don't care before disabling this check.
        raise_warnings : bool (optional, default=True)
            whether to raise warnings during conversion, related to different kinds of events we think
            the user should be aware of (e.g. images with different shapes, files overwritten, etc.).

        Raises
        ------
        ClinicaDLArgumentError
            if a json file with the same `json_name` already exists and the new conversion cannot
            be merged with the old one.
        ClinicaDLCAPSError
            if images don't have the same voxel spacing across (participant, session), and
            `ignore_spacing` is False.
        ClinicaDLCAPSError
            if some image-specific masks don't have the same shape and affine matrix as the image.

        - Also raises a warning (only once) if images have different shapes across (participant, session)
        pairs (unless `raise_warnings` is False).
        - Also raises a warning if some tensor files already present in the caps directory will be
        overwritten (unless `raise_warnings` is False).
        """

        self.tensor_conversion.convert_to_tensors(
            json_name, save_transforms, n_proc, ignore_spacing, raise_warnings
        )
        self._load_pt_masks()
        self._count_samples()

    def describe(self) -> Dict[str, Any]:
        """
        Returns a description of the CapsDataset.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - `total_samples`: the size of the dataset, i.e.
            the total number of samples.
            - `participant_session_pairs`: the list of participant/session
            pairs in the dataset.
            - `preprocessing`: the preprocessing parameters.
            - `extraction`: the extraction parameters.

        Raises
        ------
        ClinicaDLCAPSError
            if samples are extracted from the images and 'to_tensors' or
            'read_tensor_conversion' has not been run previously.
        """
        return {
            "total_samples": len(self),
            "participant_session_pairs": self.get_participant_session_couples(),
            "preprocessing": self.preprocessing.model_dump(),
            "extraction": self.extraction.model_dump(),
        }

    def get_sample_info(self, idx: int, column: str) -> Any:
        """
        Retrieves information on a given sample. The information will
        correspond to the information on the base image the sample was extracted
        from.

        Parameters
        ----------
        idx : int
            The index of the sample in the dataset.
        column : str
            The information to look for, i.e. a column of the DataFrame containing
            the metadata, which is equal to `data` if `data` was passed when instantiating the
            CapsDataset. If `data` was not passed, the only accessible columns are
            `"participant_id"` and `"session_id"`.

        Returns
        -------
        Any
            the information (e.g. the age, the sex, etc.)

        Raises
        ------
        IndexError
            If 'idx' is not a non-negative integer, greater or equal to
            the length of the dataset.
        ClinicaDLCAPSError
            if samples are extracted from the images and 'to_tensors' or
            'read_tensor_conversion' has not been run previously.
        KeyError
            If `column` is not in the metadata DataFrame.
        """
        if not isinstance(idx, int) or idx < 0:
            raise ValueError(f"Index must be a non-negative integer, got {idx}.")
        if idx >= len(self):
            raise IndexError(
                f"Index out of range, there are only {len(self)} samples in the dataset."
            )
        if column not in self.df.columns:
            raise KeyError(
                f"No column named {column} in the metadata DataFrame. Present columns are: "
                f"{list(self.df.columns)}"
            )

        row = self.df[(self.df[FIRST_INDEX] <= idx) & (idx <= self.df[LAST_INDEX])]

        return row[column].iloc[0]

    def get_participant_session_couples(self) -> List[Tuple[str, str]]:
        """
        Retrieves all participant-session pairs in the dataset.

        Returns
        -------
        List[Tuple[str, str]]
            the list of (participant, session).
        """
        return list(zip(self.df[PARTICIPANT_ID], self.df[SESSION_ID]))

    def eval(self) -> None:
        """
        Sets the dataset to evaluation mode.

        This disables data augmentation in the transformation pipeline.
        """
        self.eval_mode = True

    def train(self) -> None:
        """
        Sets the dataset to training mode.

        This enables data augmentation in the transformation pipeline.
        """
        self.eval_mode = False

    def subset(self, data: DataType) -> CapsDataset:
        """
        To get a subset of the CapsDataset from a list of (participant, session) pairs.

        Parameters
        ----------
        data : DataType
            A DataFrame (or a path to a TSV file containing the dataframe) with the list of participant/session
            pairs to extract. Please note that this list must be passed via two columns named `"participant_id`"
            and `"session_id"` (other columns won't be considered).

        Returns
        -------
        CapsDataset
            A subset of the original CapsDataset, restricted to the (participant, session) pairs mentioned in `data`.

        Raises
        ------
        ClinicaDLTSVError
            If `data` is a TSV file that does not exist.
        ClinicaDLTSVError
            If the DataFrame associated to `data` does not contain the columns `"participant_id"`
            and `"session_id"`.
        ClinicaDLTSVError
            If some (participant, session) pairs mentioned in `data` are not in the CapsDataset.
        """
        new_df = self._check_data_instance(data).set_index([PARTICIPANT_ID, SESSION_ID])

        try:
            subset_df = (
                self.df.set_index([PARTICIPANT_ID, SESSION_ID])
                .loc[new_df.index]
                .reset_index()
            )
        except KeyError as exc:
            missing_pairs = new_df.index.difference(
                self.df.set_index([PARTICIPANT_ID, SESSION_ID]).index
            )

            err_message = (
                "Some couples (participant, session) are not in the dataset:\n"
            )
            for pair in missing_pairs:
                err_message += f" - {pair} \n"
            raise ClinicaDLTSVError(err_message) from exc

        dataset = deepcopy(self)
        dataset.df = subset_df

        return dataset

    def __len__(self) -> int:
        """
        Computes the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples in the dataset, i.e. the number of images
            times the number of samples per image.

        Raises
        ------
        ClinicaDLCAPSError
            if samples are extracted from the images and 'to_tensors' or
            'read_tensor_conversion' has not been run previously.
        """
        if N_SAMPLES not in self.df.columns:
            self._count_samples()
        return int(self.df[N_SAMPLES].sum())

    def __getitem__(self, idx: int) -> Sample:
        """
        Retrieves the sample at a given index.

        Parameters
        ----------
        idx : int
            Index of the sample in the dataset.

        Returns
        -------
        Sample
            A structured output containing the processed data and metadata.

        Raises
        ------
        ClinicaDLCAPSError
            If 'to_tensors' or 'read_tensor_conversion' has not been called previously.
        ValueError
            If 'idx' is not an non-negative integer.
        IndexError
            If 'idx' is greater or equal to the length of the dataset.
        FileNotFoundError
            If the '.pt' file cannot be found for the (participant, session) associated
            to 'idx'.
        """
        if self.tensor_conversion.json is None:
            raise ClinicaDLCAPSError(
                "Cannot find tensor files. Please convert your CapsDataset "
                "to tensors using 'to_tensors', or use 'read_tensor_conversion' if it has "
                "already be done."
            )

        participant, session, sample_index = self._get_meta_data(idx)
        data = self._get_data(participant, session)
        tensor_path = self.caps_reader.get_tensor_path(
            participant, session, preprocessing=self.preprocessing, check=False
        )

        if (
            not self.tensor_conversion.get_info().transforms
        ):  # image transforms not saved
            data = self.image_transform(data)

        try:
            sample, sample_description = self.extraction.extract_sample(
                data, sample_index
            )
        except IndexError as exc:
            raise ClinicaDLCAPSError(
                f"An error occurred while extracting samples from images of ({participant}, {session})."
            ) from exc

        sample = self.sample_transform(sample)

        if not self.eval_mode:
            sample = self.augmentation(sample)

        return self.extraction.format_output(
            sample,
            image_path=tensor_path,
            description=sample_description,
        )

    ### to read user inputs ###
    def _check_label(self, label: Optional[str]) -> Optional[Union[Column, Mask]]:
        """
        Checks if 'label' is a column name, a mask suffix or None.

        Raises
        ------
        ClinicaDLArgumentError
            If 'label' is not a string or None.
        """
        if isinstance(label, str):
            if label in self.df.columns:
                if self.df[label].dtype == str:
                    label_list = self.df[label].unique()
                    if len(label_list) > 5:
                        raise ClinicaDLArgumentError(
                            f"Column '{label}' contains to many values. "
                            "It should contain maximum 5 different values for ClinicaDL to consider it as Classification"
                        )
                    else:
                        self.label_dict = {
                            key: value for key, value in enumerate(label_list)
                        }

                return Column(label)
            else:
                return self._read_mask(label)
        elif label is None:
            return None
        else:
            raise ClinicaDLArgumentError(
                f"'label' must be a string or None. Got {label}"
            )

    def _read_masks(
        self,
        masks: Optional[list[PathType]],
    ) -> tuple[list[Mask], list[Mask]]:
        """
        Reads the masks passed by the user and splits them between common masks
        and image-specific masks.

        Raises
        ------
        FileNotFoundError
            If a path is passed for a mask, and this path does not match any file.
        ClinicaDLCAPSError
            If a suffix is passed for a mask, and this suffix is a name
            among {'image', 'label', 'affine', 'participant', 'session'}.
        """
        if masks is None:
            return [], []

        if not isinstance(masks, (tuple, list)):
            raise ClinicaDLArgumentError(
                f"'masks' should be a list or a tuple, got: {masks}"
            )

        mask_objects: list[Mask] = [self._read_mask(mask) for mask in masks]

        if len(mask_objects) != len(set(mask.name for mask in mask_objects)):
            raise ClinicaDLArgumentError(
                f"Duplicated mask names in {masks}. "
                "Beware that if you passed a path in 'masks' (e.g. 'leftHippocampus.nii.gz'), "
                "CapsDatset will identify it with its file name, without "
                "the extension (e.g. 'leftHippocampus')."
            )

        for mask in mask_objects:
            if mask.name in {IMAGE, LABEL, AFFINE, PARTICIPANT, SESSION}:
                raise ClinicaDLArgumentError(
                    f"Mask suffix cannot be {mask.name}. {IMAGE, LABEL, AFFINE, PARTICIPANT, SESSION} "
                    "are protected names. Please change the suffix of your masks."
                )

        individual_masks = [mask for mask in mask_objects if not mask.is_common_mask]
        common_masks = [mask for mask in mask_objects if mask.is_common_mask]

        return individual_masks, common_masks

    def _read_mask(self, mask: PathType) -> Mask:
        """
        Determines if a mask is a common or an individual mask.
        """
        if Path(mask).suffix:  # it is a file
            return Mask(self.caps_reader.get_common_mask_path(mask))
        else:
            return Mask(mask)

    def _get_df_from_input(self, data: Optional[DataType]) -> pd.DataFrame:
        """
        Generates or validates the DataFrame from the input data.

        Raises
        ------
        ClinicaDLArgumentError
            If 'data' is not a DataFrame, a path or None.
        ClinicaDLTSVError
            If 'data' is a TSV file that does not exist.
        ClinicaDLTSVError
            If the DataFrame is empty.
        ClinicaDLTSVError
            If the DataFrame does not contain the columns `"participant_id"`
            and `"session_id"`.
        ClinicaDLTSVError
            If the DataFrame contains duplicated (participant_id, session_id) pairs.
        ClinicaDLConfigurationError
            If the data does not match the preprocessing configuration.
        """
        if data is None:
            data = self.caps_reader.create_subjects_sessions_tsv(self.preprocessing)
            print(f"Creating a TSV file at {data}")

        if not isinstance(data, (str, Path, pd.DataFrame)):
            raise ClinicaDLArgumentError(
                f"'data' must be a Pandas DataFrame, a path to a TSV file or None. Got {data}"
            )

        df = self._check_data_instance(data)

        return deepcopy(df)

    @staticmethod
    def _check_data_instance(data: DataType) -> pd.DataFrame:
        """
        Checks the DataFrame passed by the user (either as a DataFrame or
        as a path to a TSV). Returns the checked DataFrame.
        """
        if isinstance(data, (str, Path)):
            path = Path(data)
            df = tsv_to_df(path)
        elif isinstance(data, pd.DataFrame):
            df = check_df(data)

        return df  # pylint: disable=possibly-used-before-assignment

    ### for __getitem__ ###
    def _get_meta_data(self, idx: int) -> Tuple[str, str, int]:
        """
        Retrieves metadata for a given index.
        'idx' is the index of the sample in the dataset.

        Returns
        -------
        tuple
            - participant (str): ID of the participant.
            - session (str): ID of the session.
            - sample_index (int): index of the extracted sample
            in the original image.

        Raises
        ------
        IndexError
            If 'idx' is out of range.
        """

        # img_idx = idx // self.samples_per_image

        participant = self.get_sample_info(idx, PARTICIPANT_ID)

        session = self.get_sample_info(idx, SESSION_ID)
        row = self.df.set_index([PARTICIPANT_ID, SESSION_ID]).loc[
            (participant, session)
        ]
        sample_idx = int(idx - row.at[FIRST_INDEX])

        # sample_idx = idx % row.at[N_SAMPLES]

        return participant, session, sample_idx

    def _get_data(self, participant: str, session: str) -> DataPoint:
        """
        Gets all the images relevant to the (participant, session)
        i.e. the image and the masks, individual and common.

        Conversion to tensors must have been performed first.

        Raises
        ------
        FileNotFoundError
            If the '.pt' file cannot be found for this (participant, session).
        """
        pt_path = self.caps_reader.get_tensor_path(
            participant, session, self.preprocessing, check=False
        )
        images_dict = self._load_pt(pt_path)

        # label
        if isinstance(self.label, Mask):
            label_mask = images_dict[self.label.name]
            label = tio.LabelMap(tensor=label_mask, affine=images_dict[AFFINE])
        else:
            label = self._get_scalar_label(participant, session)

        data = DataPoint(
            image=tio.ScalarImage(
                tensor=images_dict[IMAGE], affine=images_dict[AFFINE]
            ),
            label=label,
            participant=participant,
            session=session,
        )

        individual_masks_name = [mask.name for mask in self.individual_masks]
        # individual masks
        for name, image in images_dict.items():
            if name not in {IMAGE, LABEL, AFFINE} and name in individual_masks_name:
                data.add_mask(
                    tio.LabelMap(tensor=image, affine=images_dict[AFFINE]), name
                )

        # common masks (already loaded)
        for mask in self.common_masks_tensors:
            data.add_mask(mask.get_associated_mask(), mask.name)

        return data

    def _load_pt(self, path: Path) -> Dict[str, Any]:
        """
        Loads the tensors for a (participant, session).
        See also: :py:func:`clinicadl.data.tensor_conversion.TensorConversion.save_images_as_tensors`
        """
        try:
            return torch.load(path, weights_only=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Tensor conversion was performed, as suggested in '{self.tensor_conversion.json}'. "
                f"Nevertheless, file '{str(path)}'  cannot be found. The tensors have probably been deleted "
                "after conversion. Please rerun 'to_tensors' to generate the tensor files again."
            ) from exc

    def _get_scalar_label(
        self, participant: str, session: str
    ) -> Optional[Union[int, float]]:
        """
        Returns the label when it is not an image.
        """
        if self.label is None:
            return None
        elif isinstance(self.label, Column):
            df_tmp = self.df.set_index([PARTICIPANT_ID, SESSION_ID])
            if df_tmp[self.label._name].dtype == str:
                return self.label_dict[df_tmp.at[(participant, session), self.label]]
            return df_tmp.at[(participant, session), self.label]

    ### other utils ###
    def _load_pt_masks(self) -> None:
        """
        Converts nifti masks to the associated tensor masks
        when 'to_tensors' or 'read_tensor_conversion' is called.
        """
        self.common_masks_tensors = []
        for mask in self.common_masks:
            mask_pt_path = self.caps_reader.path_to_tensor(mask.path)
            try:
                self.common_masks_tensors.append(Mask(mask_pt_path))
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Tensor conversion was performed, as suggested in '{self.tensor_conversion.json}'. "
                    f"Nevertheless, mask '{str(mask_pt_path)}' cannot be found. The tensors have probably been deleted "
                    "after conversion. Please rerun 'to_tensors' to generate the tensor file again."
                ) from exc

    def _count_samples(self) -> None:
        """
        Gets the number of samples for each image and puts
        it in the dataframe.
        """
        if self.extraction.extract_method == ExtractionMethod.IMAGE:
            self.df[N_SAMPLES] = 1
        else:
            if self.tensor_conversion.json is None:
                raise ClinicaDLCAPSError(
                    "Needs tensors to compute the length of the dataset (which depends "
                    "on the number of samples per image). Please convert your CapsDataset "
                    "to tensors using 'to_tensors', or use 'read_tensor_conversion' if it has "
                    "already be done."
                )
            if (
                self.tensor_conversion.get_info().shape
            ):  # uniform shape across the dataset
                first_row = self.df.iloc[0]
                participant, session = first_row[PARTICIPANT_ID], first_row[SESSION_ID]
                self.df[N_SAMPLES] = self._get_n_samples(participant, session)
            else:
                for idx, row in self.df.iterrows():
                    participant = row[PARTICIPANT_ID]
                    session = row[SESSION_ID]
                    self.df.at[idx, N_SAMPLES] = self._get_n_samples(
                        participant, session
                    )

        self._map_indices_to_images()

    def _get_n_samples(self, participant: str, session: str) -> int:
        """
        Gets the number of samples in an image.
        """
        data = self._get_data(participant, session)
        if (
            not self.tensor_conversion.get_info().transforms
        ):  # image transforms not saved
            data = self.image_transform(data)
        try:
            return self.extraction.num_samples_per_image(data.image.tensor)
        except IndexError as exc:
            raise ClinicaDLCAPSError(
                f"An error occurred while counting samples in images of ({participant}, {session})."
            ) from exc

    def _map_indices_to_images(self) -> None:
        """
        To have in the dataframe the last and the first sample index
        corresponding to each image.
        """
        self.df[FIRST_INDEX] = (
            (self.df[N_SAMPLES].cumsum().shift(1)).fillna(0).astype(int)
        )
        self.df[LAST_INDEX] = (self.df[N_SAMPLES].cumsum() - 1).astype(int)
