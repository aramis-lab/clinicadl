from typing import Generic, Optional, TypeVar

import torchio as tio

from clinicadl.io import Bids, BidsFileType
from clinicadl.utils.bids import BidsFile
from clinicadl.utils.typing import PathType

ImageT = TypeVar("T", bound=tio.Image)


class _SubjectSpecificImage(Generic[ImageT]):
    image_type: ImageT

    def __init__(self, bids: Bids, file_type: BidsFileType):
        self.bids = bids
        self.file_type = file_type

    def get(self, participant: str, session: str) -> ImageT:
        """
        Loads the image associated to the (participant, session) pair.

        Parameters
        ----------
        participant : str
            The participant id (e.g., "sub-xxx").
        session : str
            The session id (e.g., "ses-xxx").

        Returns
        -------
        tio.Image
            The image in a :py:class:`torchio.Image`.
        """
        path = self.bids.get_path(
            self.file_type, participant=participant, session=session
        )

        return self.image_type(path=path)


class Image(_SubjectSpecificImage[tio.ScalarImage]):
    """
    To handle image loading from a :term:`BIDS` given a :py:class:`~clinicadl.io.BidsFileType`.
    """

    image_type = tio.ScalarImage


class IndividualMask(_SubjectSpecificImage[tio.LabelMap]):
    """
    To handle image-specific mask loading from a :term:`BIDS` given a :py:class:`~clinicadl.io.BidsFileType`.
    """

    image_type = tio.LabelMap


class CommonMask:
    """
    To handle non-image-specific mask loading.
    """

    def __init__(self, path: PathType):
        self.file = BidsFile(path)
        self._mask: Optional[tio.LabelMap] = None  # lazy loading

    def get(self) -> tio.LabelMap:
        """
        Returns
        -------
        tio.LabelMap
            The mask in a :py:class:`torchio.LabelMap`.
        """
        if not self._mask:
            self._mask = tio.LabelMap(path=self.file.path)

        return self._mask
