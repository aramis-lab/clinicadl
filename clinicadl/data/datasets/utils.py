from __future__ import annotations

from abc import abstractmethod
from bisect import bisect_right
from copy import deepcopy
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Optional,
    Sequence,
    TypeAlias,
    Union,
)

import pandas as pd
from tqdm import tqdm

from clinicadl.utils.dictionary.words import (
    N_SAMPLES,
    PARTICIPANT_ID,
    SAMPLE_TYPE,
    SESSION_ID,
)
from clinicadl.utils.tsvtools import read_data
from clinicadl.utils.typing import DataFrameType

from ..structures.sample import SAMPLE_FIELDS, Sample, Sample2D, SampleType
from ..utils import DEFAULT_SPATIAL_CHECKS, DatasetChecker, SpatialCheck
from .base import Dataset

if TYPE_CHECKING:
    from clinicadl.transforms import TransformsHandler

    from ..structures import DataPoint

logger = getLogger(__name__)


class _MultiSamplesDataset(Dataset[Sample]):
    """
    An abstract :py:class:`~clinicadl.data.datasets.Dataset` to handle multiple samples per image.

    Here the size of the dataset is not equal to the number of images, since an image can contain multiple samples
    (e.g. patches or slices).

    The key idea of this dataset is that a column of the metadata DataFrame is expected to have a column named "n_samples", with
    the number of samples for each image. If the DataFrame doesn't have this column, ``SamplesDataset`` will try to call
    :py:meth:`_count_samples`.
    """

    @property
    def _has_len(self) -> bool:
        """Whether the number of samples for each image is known."""
        return N_SAMPLES in self.df.columns

    def get_sample_info(self, idx: int, column: str) -> Any:
        self._check_has_len()
        self._check_idx(idx)
        self._check_column(column)

        image_idx = self._get_image_idx(idx)
        row = self.df.iloc[image_idx]

        value = row.at[column]
        try:
            return value.item()  # e.g. convert np.int to int
        except AttributeError:
            return value

    def __len__(self) -> int:
        self._check_has_len()

        return int(self.df[N_SAMPLES].sum())

    def _check_has_len(self) -> None:
        """
        Checks that the length of the dataset has been computed,
        otherwise tries to compute it with :py:meth:`_count_samples`.
        """
        if not self._has_len:
            self._count_samples()

    def _get_image_idx(self, idx: int) -> int:
        """
        Gets the image in the dataset corresponding to the index
        of the sample.
        """
        return bisect_right(self.df[N_SAMPLES].cumsum(), idx)

    def _get_index_in_image(self, idx: int) -> int:
        """
        Determines the the index of this sample in its original image.

        E.g.: if every image has 3 samples, then self._get_index_in_image(4)=1,
        because the first image contains samples 0, 1 and 2, and the second
        image contains the samples 3, 4 and 5, so 4 is the 2nd sample
        of its image.
        """
        image_idx = self._get_image_idx(idx)
        if image_idx > 0:
            return int(idx - self.df[N_SAMPLES].cumsum().iat[image_idx - 1])

        return idx

    def _get_image_info(self, participant: str, session: str, column: str) -> Any:
        """
        Returns the value of a column for a (participant, session).
        """
        self._check_column(column)

        return self.df.set_index([PARTICIPANT_ID, SESSION_ID]).at[
            (participant, session), column
        ]

    def _get_indices_associated_to(
        self, participant: str, session: str
    ) -> tuple[int, int]:
        """
        Returns the value of the range of indices associated to the image.
        """
        self._check_has_len()

        cumsum = self.df.set_index([PARTICIPANT_ID, SESSION_ID])[N_SAMPLES].cumsum()
        min_idx = cumsum.shift(1).fillna(0).at[(participant, session)]
        max_idx = max(cumsum.at[(participant, session)] - 1, 0)

        return int(min_idx), int(max_idx)

    def _count_samples(self) -> None:
        """
        Gets the number of samples for each image and puts
        it in the metadata DataFrame in the column 'n_samples'.
        """
        raise NotImplementedError(
            "_count_samples must be implemented if there is no column named 'n_samples' in the metadata DataFrame."
        )


class SamplerDataset(_MultiSamplesDataset):
    """
    An abstract :py:class:`~clinicadl.data.datasets.Dataset` that can sample 3D patches or 2D slices from
    a 3D image.

    It inherits from :py:class:`MultiSamplesDataset`, so the length of the dataset depends
    on the number of samples in each image, which is expected to be given in the metadata DataFrame or calculated with
    the method :py:meth:`_count_samples`.

    This dataset also deals with the transformation pipeline to apply to the data, with a distinction between the transformations
    apply to the whole 3D images, and those apply to the sample (e.g. a patch or a slice). See :py:class:`clinicadl.transforms.TransformsHandler`.
    """

    def __init__(self, transforms: TransformsHandler):
        self.eval_mode = False
        self.transforms = transforms

    def eval(self) -> None:
        self.eval_mode = True

    def train(self) -> None:
        self.eval_mode = False

    def sort(self) -> None:
        """
        Sorts the dataset by (participant, session) pairs (alphabetic order).

        Examples
        --------
        .. code-block:: python
            >>> dataset[0].participant
            'sub-001'
            >>> dataset[1].participant
            'sub-000'
            >>> dataset.sort()
            >>> dataset[0].participant
            'sub-000'
        """
        self.df.sort_values([PARTICIPANT_ID, SESSION_ID], inplace=True)

    def __getitem__(self, idx: int) -> Sample:
        participant, session, index_in_image = self._get_sample_meta_data(idx)
        data = self._get_data(participant, session)

        data = self.transforms.apply_image_transforms(data)

        data = self.transforms.extract_sample(data, index_in_image)

        data = self.transforms.apply_sample_transforms(data)

        if not self.eval_mode:
            data = self.transforms.apply_augmentations(data)

        return self._format_output(data)

    def _get_sample_meta_data(self, idx: int) -> tuple[str, str, int]:
        """
        Retrieves the metadata for a given index.
        ``idx`` is the index of the sample in the dataset.
        """
        participant = self.get_sample_info(idx, PARTICIPANT_ID)
        session = self.get_sample_info(idx, SESSION_ID)
        index_in_image = self._get_index_in_image(idx)

        return participant, session, index_in_image

    @abstractmethod
    def _get_data(self, participant: str, session: str) -> DataPoint:
        """
        Returns that data for a (participant, session) in a :py:class:`~clinicadl.data.structures.DataPoint`.

        Parameters
        ----------
        participant : str
            The id of the participant.
        session : str
            The id of the session.

        Returns
        -------
        clinicadl.data.structures.DataPoint
            The data associated to the (participant, session), with at least the image but also potential
            metadata that should be in the output of the dataset or that are useful in the transformation
            pipeline.
        """

    def _format_output(self, output: DataPoint) -> Sample:
        """
        Formats the output depending on the type of sample.
        """
        if self.transforms.extraction.sample_type == SampleType.SLICE:
            del output[SAMPLE_TYPE]
            return Sample2D(**output)

        return Sample(**output)

    def _count_samples(self) -> None:
        if self.transforms.extraction.sample_type == SampleType.IMAGE:
            self._df[N_SAMPLES] = 1
        else:
            for idx, row in tqdm(
                self._df.iterrows(),
                desc="Counting the number of samples per image",
                unit="images",
            ):
                participant = row[PARTICIPANT_ID]
                session = row[SESSION_ID]
                try:
                    self._df.at[idx, N_SAMPLES] = self._count_in_image(
                        participant, session
                    )
                except Exception as e:
                    e.add_note(
                        f"\nAn error occurred when reading the data of ({participant}, {session}) "
                        "to count the number of samples (see above)."
                    )
                    raise

    def _count_in_image(self, participant: str, session: str) -> int:
        """
        Gets the number of samples in an image.
        """
        data = self._get_data(participant, session)

        data = self.transforms.apply_image_transforms(data)

        return self.transforms.extraction.num_samples_per_image(data)


ColumnsType: TypeAlias = Union[
    Sequence[str], dict[str, Optional[Callable[[pd.Series], pd.Series]]]
]


class MultimodalSamplerDataset(SamplerDataset):
    """
    Child of :py:class:`SamplerDataset` that can add metadata to the output sample.
    """

    def __init__(
        self,
        data: DataFrameType,
        transforms: TransformsHandler,
        columns: Optional[ColumnsType],
    ):
        super().__init__(transforms)

        self._df = self._validate_df(data)
        columns = self._validate_columns(columns)
        self._process_columns(columns)

        self.columns = list(columns.keys())

    def _validate_df(self, data: DataFrameType) -> pd.DataFrame:
        """
        Validates the input DataFrame.
        """
        df = read_data(data)

        return deepcopy(df)

    def _validate_columns(
        self,
        columns: Optional[
            Union[Sequence[str], dict[str, Optional[Callable[[pd.Series], pd.Series]]]]
        ],
    ) -> dict[str, Optional[Callable[[pd.Series], pd.Series]]]:
        """
        Harmonises column inputs.
        """
        if columns is None:
            columns = dict()

        if isinstance(columns, Sequence):
            columns = {col: None for col in columns}

        return self._check_keys(columns, "column")

    @staticmethod
    def _check_keys(
        dict_: dict[str, Any],
        dict_type: str,
    ) -> dict[str, Any]:
        """
        Checks column names.
        """
        for col in dict_:
            if col in SAMPLE_FIELDS:
                raise ValueError(
                    f"A {dict_type} cannot be named '{col}'. {SAMPLE_FIELDS} "
                    "are protected names."
                )

        return dict_

    def _process_columns(
        self,
        columns: dict[str, Optional[Callable[[pd.Series], pd.Series]]],
    ) -> None:
        """
        Processes the input DataFrame with encoding functions passed by the user.
        """
        for column, encoding in columns.items():
            self._check_column(column)
            if encoding is None:
                continue
            try:
                self.df[column] = encoding(self.df[column])
            except Exception as e:
                raise RuntimeError(
                    f"Unable to process the column '{column}' with the function you passed. "
                    "Make sure that this function takes as input a Pandas Series, and returns a Pandas Series."
                ) from e

    def _get_data(self, participant: str, session: str) -> DataPoint:
        """
        Returns that data for a (participant, session) in a DataPoint.
        """
        datapoint = self._get_images(participant, session)

        for col in self.columns:
            datapoint[col] = self._get_image_info(participant, session, col)

        return datapoint

    @abstractmethod
    def _get_images(self, participant: str, session: str) -> DataPoint:
        """
        Loads the image and the masks.
        """


class CheckableDataset(Dataset):
    def sanity_check(
        self,
        spatial_checks: Optional[Iterable[str | SpatialCheck]] = DEFAULT_SPATIAL_CHECKS,
    ) -> None:
        """
        Performs a sanity check on the current dataset.

        It will iterate over the whole dataset to check if images are loaded and transformed correctly,
        and potentially perform spatial checks on the loaded images.

        Parameters
        ----------
        spatial_checks : Optional[Iterable[str  |  SpatialCheck]], default=[ "affine", "shape", "global_spacing"]
            Spatial checks to perform on the images:

            - ``"spacing"``: checks **intra-sample voxel spacing consistency**, i.e. that all the images and masks
              in a :py:class:`~clinicadl.data.structures.Sample` have the same voxel spacing.
            - ``"affine"``: checks **intra-sample affine matrix consistency** (so it includes ``"spacing"``).
            - ``"shape"``: checks **intra-sample spatial shape consistency**.
            - ``"global_spacing"``: checks **inter-sample voxel spacing consistency**, i.e. that all the ``Samples``
              in the dataset have the same voxel spacing (so it includes ``"spacing"``).
            - "``global_shape"``: checks **inter-sample spatial shape consistency** (so it includes ``"shape"``).

            If ``None``, no spatial check.
        """
        DatasetChecker(spatial_checks).check(self)
