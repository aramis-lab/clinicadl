import abc

from pydantic import computed_field

from clinicadl.utils.config import ClinicaDLConfig

from ..file_type import FileType


class RawData(ClinicaDLConfig, abc.ABC):
    """
    Abstract base class to handle raw (unprocessed) neuroimaging data in BIDS format.
    """

    @computed_field
    @property
    def file_type(self) -> FileType:
        """
        The file type associated with the BIDS dataset.

        This property uses the `_get_bids_filetype` method to return the correct
        file type for the modality being handled.
        """
        return self._get_bids_filetype()

    @abc.abstractmethod
    def _get_bids_filetype(self) -> FileType:
        """
        Abstract method to obtain the BIDS-compatible FileType.

        This method must be implemented by subclasses to specify the file type
        associated with the modality.
        """

    def __str__(self):
        """
        String description of the data.
        """
        return self.file_type.description
