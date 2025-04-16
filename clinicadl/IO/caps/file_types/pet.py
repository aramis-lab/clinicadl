from typing import Optional, Union

from ...file_type import FileType
from ...modalities.pet import PET, ReconstructionMethod, Tracer


class PETFileType(FileType, PET):
    """
    Configuration class to handle raw custom imaging data with a user-defined suffix.
    """

    def __init__(
        self,
        tracer: Union[str, Tracer],
        reconstruction: Optional[Union[ReconstructionMethod, str]] = None,
    ):
        """
        Initialize the PETFileType with tracer and optional reconstruction method.

        Args:
            tracer (str): The tracer used in the PET scan.
            reconstruction (Optional[str]): The reconstruction method used. Defaults to None.
        """
        self.tracer = Tracer(tracer)
        self.reconstruction = ReconstructionMethod(reconstruction)

        trc = f"_trc-{self.tracer}"
        rec = ""
        if self.reconstruction:
            rec = f"_rec-{self.reconstruction}"

        super().__init__(pattern=trc + rec)

    @property
    def description(self) -> str:
        """
        The description of the file type.
        """
        description = f"Raw PET NIfTI images with tracer '{self.tracer}'"
        if self.reconstruction:
            description += f" and reconstruction method '{self.reconstruction}'"
        return description

    @property
    def container(self) -> str:
        """
        The name of the folder where the file is stored.
        """
        return self.modality
