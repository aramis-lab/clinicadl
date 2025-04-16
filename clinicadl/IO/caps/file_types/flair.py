from clinicadl.dictionary.words import ANAT

from ...file_type import FileType
from ...modalities import Flair


class FlairFileType(FileType, Flair):
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
        return "Raw FLAIR T2w MRI NIfTI images"

    @property
    def container(self) -> str:
        """
        The name of the folder where the file is stored.
        """
        return ANAT
