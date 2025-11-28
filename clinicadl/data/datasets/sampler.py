from abc import abstractmethod
from typing import Optional

from tqdm import tqdm

from clinicadl.dictionary.words import (
    N_SAMPLES,
    PARTICIPANT_ID,
    SAMPLE_TYPE,
    SESSION_ID,
)
from clinicadl.transforms.handlers import Transforms

from ..structures import DataPoint
from .multi_samples import MultiSamplesDataset
from .output import Sample, Sample2D, SampleType


class SamplerDataset(MultiSamplesDataset):
    """
    An abstract :py:class:`~clinicadl.data.datasets.ClinicaDLDataset` that can sample 3D patches or 2D slices from
    a 3D image.

    It inherits from :py:class:`~clinicadl.data.datasets.MultiSamplesDataset`, so the length of the dataset depends
    on the number of samples in each image, which is expected to be given in the metadata DataFrame or calculated with
    the method :py:meth:`_count_samples`.

    This dataset also deals with the transformation pipeline to apply to the data, with a distinction between the transformations
    apply to the whole 3D images, and those apply to the sample (e.g. a patch or a slice). See :py:class:`clinicadl.transforms.Transforms`.

    See Also
    --------
    clinicadl.data.datasets.MultiSamplesDataset
    """

    eval_mode: bool
    transforms: Transforms
    _initial_shape: Optional[
        tuple[int, int, int, int]
    ] = None  # the shape (C, W, H, D) of the image before any transformation (if it is consistent across the dataset)

    def eval(self) -> None:
        self.eval_mode = True

    def train(self) -> None:
        self.eval_mode = False

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
        index_in_image = self._get_rank_in_row(idx)

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
            if self._initial_shape:  # uniform shape across the dataset
                first_row = self._df.iloc[0]
                participant, session = first_row[PARTICIPANT_ID], first_row[SESSION_ID]
                self._df[N_SAMPLES] = self._count_in_image(participant, session)
            else:
                for idx, row in tqdm(
                    self._df.iterrows(),
                    desc="Counting the number of samples per image",
                    unit="images",
                ):
                    participant = row[PARTICIPANT_ID]
                    session = row[SESSION_ID]
                    self._df.at[idx, N_SAMPLES] = self._count_in_image(
                        participant, session
                    )

    def _count_in_image(self, participant: str, session: str) -> int:
        """
        Gets the number of samples in an image.
        """
        data = self._get_data(participant, session)

        data = self.transforms.apply_image_transforms(data)

        return self.transforms.extraction.num_samples_per_image(data)
