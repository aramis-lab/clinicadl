from ...file_type import FileType
from ...modalities import DWI


class DWIFileType(FileType, DWI):
    """
    Configuration class to handle raw custom imaging data with a user-defined suffix.
    """

    def __init__(self):
        """
        Generate the BIDS-compatible file type pattern and description.

        Returns:
            FileType: A FileType object containing the pattern and description.
        """

        super().__init__()

    @property
    def description(self) -> str:
        """
        The description of the file type.
        """
        return "Raw DW MRI NIfTI images"

    @property
    def container(self) -> str:
        """
        The name of the folder where the file is stored.
        """
        return self.modality
