from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from pydantic import field_validator

from clinicadl.io.bids import Bids
from clinicadl.transforms import TransformsHandler
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig
from clinicadl.utils.typing import DataFrameType, PathType

from ..structures import Tensor
from ..tensors import TensorDescription
from .bids_utils import (
    BidsTensorDataset,
    BidsTypeDatasetConfig,
    BidsTypeDatasetWithConfig,
    ColumnsType,
)


class TensorDatasetConfig(ObjectConfig["TensorDataset"], BidsTypeDatasetConfig):
    """Config class to check ``TensorDataset`` inputs."""

    description_json: Path
    to_load: Optional[Sequence[str]]

    @field_validator("description_json", mode="after")
    @classmethod
    def _resolve_path(cls, v: Path) -> Path:
        return v.resolve()

    @classmethod
    def _get_class(cls):
        return TensorDataset


class TensorDataset(
    BidsTensorDataset, HasConfig[TensorDatasetConfig], BidsTypeDatasetWithConfig
):
    """
    A :py:class:`~clinicadl.data.datasets.Dataset` to read images saved as tensors in
    ``.pt`` files.

    This dataset enables to load tensors saved with
    :py:meth:`BidsDataset.to_tensors <clinicadl.data.datasets.BidsDataset.to_tensors>`.

    Parameters
    ----------
    description_json : PathType
        The path to the ``.json`` file saved by :py:meth:`BidsDataset.to_tensors <clinicadl.data.datasets.BidsDataset.to_tensors>`
        that describes the conversion.

    data : Optional[DataFrameType], default=None
        A :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the DataFrame) with the list of (participant, session)
        pairs to consider, as well as any other relevant information (e.g. the age of the participants). Only (participant, session)
        pairs mentioned in this TSV file will be in the ``TensorDataset``.

        If ``None``, all (participant, session) pairs whose images have been converted during this conversion will be considered.

        .. warning::
            Be careful if you pass a DataFrame with a column named ``"n_samples"``. ``BidsDataset`` will understand it as the
            number of samples for each (participant, session) pair.

    transforms : TransformsHandler, default=TransformsHandler()
        Transformation pipeline to apply to the data after loading. The user also specifies here whether to work on images, patches, or slices.
        See :py:class:`clinicadl.transforms.TransformsHandler`.

        .. warning::
            If transformed images were saved in the ``.pt`` files, make sure that you don't apply these transforms again here (``image_transforms``
            should probably be empty in the ``TransformsHandler`` here).

    columns : Optional[ColumnsType], default=None
        Columns to get in the DataFrame ``data`` and to put in the output :py:class:`~clinicadl.data.structures.Sample`.

        Can be passed via:

        - a list of strings (e.g. ``["age", "sex"]``), corresponding to the names of the columns;
        - or a dictionary (e.g. ``{"age": <function>, "sex": None}``), where the keys are the names of the columns, and the values
          are functions to apply to the columns. If the function is ``None``, no function will be applied to the column.

        .. note::
            The potential functions applied to the columns are applied to the **whole column**. They must take as input
            a :py:class:`pandas.Series`, and return a :py:class:`pandas.Series`. For example, it is useful to convert
            string labels to integer labels for classification.

    to_load : Optional[Sequence[str]], default=None
        The data to load from the ``.pt`` files. Data saved in this files are described in the descriptive ``.json`` file.
        If ``None``, everything inside the files will be loaded.

    Examples
    --------

    .. code-block:: text

        bids
        ├── dataset_description.json
        ├── metadata.tsv
        ...
        └── derivatives
            └── tensors
                ├── dataset_description.json
                ├── conversions.tsv
                ├── src-T1w_conv-T1WithMasks_description.json
                ├── src-T1w_conv-T1WithMasks_participantsXsessions.tsv
                ├── sub-001
                │   ├── ses-M000
                │   │   └── anat
                │   │       ├── sub-001_ses-M000_src-T1w_conv-T1WithMasks_tensors.json
                │   │       └── sub-001_ses-M000_src-T1w_conv-T1WithMasks_tensors.pt    <- contains the image + 2 masks named 'head' adn 'mni'
                │   ...
                ...

        The "metadata.tsv" file looks like:

        participant_id  session_id   age   sex   diagnosis
        sub-001         ses-M000     55.0  M     control
        sub-001         ses-M024     57.0  M     control
        sub-002         ses-M000     62.0  F     control
        sub-002         ses-M024     64.0  F     patient
        sub-003         ses-M000     67.0  F     patient
        ...

    .. code-block::

        from clinicadl.data.datasets import TensorDataset
        from clinicadl.transforms import TransformsHandler, extraction
        import pandas as pd


        # to convert diagnosis to numeric values
        def diagnosis_to_number(column: pd.Series) -> pd.Series:
            encoding = {"CN": 0, "MCI": 1, "AD": 2}
            return column.apply(lambda x: encoding[x])

    .. code-block::

        >>> dataset = TensorDataset(
                description_json="bids/derivatives/tensors/src-T1w_conv-T1WithMasks_description.json",
                data="bids/metadata.tsv",
                columns=["age"],
            )
        >>> dataset[0]
        Sample(Keys: ('head', 'mni', 'age', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 3)
        >>> dataset[0].spatial_shape
        (169, 208, 179)

    .. code-block::

        >>> dataset = TensorDataset(
                description_json=bids / "derivatives" / "tensors" / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_description.json",
                transforms=TransformsHandler(
                    extraction=extraction.Patch(patch_size=2),
                ),
                to_load=["head"],
            )
        >>> dataset[0]
        Sample(Keys: ('head', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 3)
        >>> dataset[0].spatial_shape
        (64, 64, 64)

    See Also
    --------
    :py:class:`clinicadl.data.datasets.BidsDataset`
    """

    _config_type = TensorDatasetConfig

    def __init__(
        self,
        description_json: PathType,
        data: Optional[DataFrameType] = None,
        transforms: TransformsHandler = TransformsHandler(),
        columns: Optional[ColumnsType] = None,
        to_load: Optional[Sequence[str]] = None,
    ):
        self.config = self._config_type(
            description_json=description_json,
            data=data,
            transforms=transforms,
            columns=columns,
            to_load=to_load,
        )
        tensors = TensorDescription.read(self.config.description_json)
        super().__init__(
            tensor=Tensor(
                Bids(self.config.description_json.parent),
                tensors.tensor_type,
                to_load=self.config.to_load,
            ),
            data=self.config.data,
            transforms=self.config.transforms,
            columns=self.config.columns,
        )
