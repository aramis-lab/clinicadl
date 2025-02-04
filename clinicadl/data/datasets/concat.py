# coding: utf8
from logging import getLogger
from typing import Any, Iterable, List, Tuple

import numpy as np
from torch.utils.data import ConcatDataset as TorchConcatDataset

from clinicadl.utils.exceptions import ClinicaDLCAPSError
from clinicadl.utils.typing import DataType

from .caps_dataset import CapsDataset, CapsDescription

logger = getLogger("clinicadl.data.datasets.concat")


# TODO : finish
class ConcatDataset(TorchConcatDataset):
    def __init__(self, datasets: Iterable[CapsDataset], ignore_spacing: bool = False):
        super().__init__(datasets)
        self.datasets: list[CapsDataset]
        self._check_consistency(ignore_spacing)

    def describe(self) -> list[CapsDescription]:
        """
        Returns a description of the CapsDataset.

        Returns
        -------
        CapsDescription
            A dataclass containing:
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
        return [dataset.describe() for dataset in self.datasets]

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

    def get_participant_session_couples(self) -> List[Tuple[str, str]]:
        """
        Retrieves all participant-session pairs in the dataset.

        Returns
        -------
        List[Tuple[str, str]]
            the list of (participant, session).
        """

    def eval(self) -> None:
        """
        Sets the dataset to evaluation mode.

        This disables data augmentation in the transformation pipeline.
        """
        for dataset in self.datasets:
            dataset.eval_mode = True

    def train(self) -> None:
        """
        Sets the dataset to training mode.

        This enables data augmentation in the transformation pipeline.
        """
        for dataset in self.datasets:
            dataset.eval_mode = False

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

    def _check_consistency(self, ignore_spacing: bool) -> None:
        """
        Checks if voxel spacing is consistent across datasets.
        """
        for dataset in self.datasets:
            if dataset.tensor_conversion.json is None:
                raise ClinicaDLCAPSError(
                    "Tensor conversion must be performed BEFORE concatenation. Please call "
                    "'to_tensors' or 'read_tensor_conversion' for each dataset."
                )

        if not ignore_spacing:
            ref_spacing = None
            ref_idx = None
            for i, dataset in enumerate(self.datasets):
                spacing = dataset.tensor_conversion.get_info().spacing
                if spacing is not None:
                    if ref_spacing is None:
                        ref_spacing = spacing
                        ref_idx = i
                    else:
                        if not np.isclose(spacing, ref_spacing, rtol=1e-2).all():
                            raise ClinicaDLCAPSError(
                                "Different voxel spacings found across datasets: "
                                f"voxel spacing is {ref_spacing} in dataset {ref_idx}, "
                                f"but {spacing} in dataset {i}.\n"
                                "If you don't care about voxel spacing, set `ignore_spacing` "
                                "to True to ignore this error."
                            )
