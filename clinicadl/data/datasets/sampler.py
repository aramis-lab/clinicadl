from abc import abstractmethod

from clinicadl.dictionary.words import SAMPLE_TYPE
from clinicadl.transforms.handlers import Transforms

from ..structures import DataPoint
from .multi_samples import MultiSamplesDataset
from .output import Sample, Sample2D, SampleType


class SamplerDataset(MultiSamplesDataset):
    """
    An abstract :py:class:`~clinicadl.data.datasets.ClinicaDLDataset` that can sample 3D patches or 2D slices from
    a 3D image.

    It inherits from :py:class:`~clinicadl.data.datasets.MultiSamplesDataset`, so the length of the dataset depends
    on the number of samples in each image.

    This dataset also deals with the transformation pipeline to apply to the data, with a distinction between the transformations
    apply to the whole 3D images, and those apply to the sample (e.g. a patch or a slice). See :py:class:`clinicadl.transforms.Transforms`.
    """

    eval_mode: bool
    transforms: Transforms

    def eval(self) -> None:
        self.eval_mode = True

    def train(self) -> None:
        self.eval_mode = False

    def __getitem__(self, idx: int) -> Sample:
        participant, session, sample_index = self._get_sample_meta_data(idx)
        data = self._get_data(participant, session)

        data = self.transforms.apply_image_transforms(data)

        data = self.transforms.extract_sample(data, sample_index)

        data = self.transforms.apply_sample_transforms(data)

        if not self.eval_mode:
            data = self.transforms.apply_augmentations(data)

        return self._format_output(data)

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
