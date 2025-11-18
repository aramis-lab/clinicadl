import abc
import os
import re
from typing import Optional, Pattern, Self, Union

from pydantic import model_validator

from ..base import DataType
from ..modalities import Modality


class Preprocessing(DataType, abc.ABC):
    """
    Abstract class to represent preprocessings.
    """

    pattern: Optional[Union[str, Pattern]] = None  # None is only for initialization
    key: Optional[str] = None

    @model_validator(mode="after")
    def _init_pattern_and_description(self) -> Self:
        """Computes the field values, AFTER initialization."""
        self.__dict__["pattern"] = self._get_pattern()
        self.__dict__["description"] = self._get_description()
        self.__dict__["key"] = self._pipeline_name

        return self

    @property
    @abc.abstractmethod
    def _pipeline_name(self) -> str:
        """The preprocessing method being applied (e.g. "t1-linear", "pet-linear")."""

    @abc.abstractmethod
    def _get_pattern(self) -> Pattern:
        """
        To obtain the file pattern associated to the preprocessing.
        """

    @abc.abstractmethod
    def _get_description(self) -> str:
        """
        To obtain a description of the preprocessing.
        """


class _LinearPreprocessing(Preprocessing, Modality):
    """
    Base class for linear preprocessings (``t1-linear``, ``flair-linear`` or ``pet-linear``).

    If the ``use_uncropped_image`` is set to ``True``, it uses the uncropped image pattern;
    otherwise, it adds the ``_desc-Crop`` suffix to the pattern to select cropped images.
    """

    use_uncropped_image: bool = False

    @property
    def _filename(self) -> str:
        return f"{self._pipeline_name}{'' if self.use_uncropped_image else '_cropped'}"

    def _get_pattern(self) -> Pattern:
        desc_crop = "" if self.use_uncropped_image else "_desc-Crop"
        file_pattern = f"sub-.*_ses-.*_space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_{self._modality}.nii.*"
        pattern = os.path.join(self._pipeline_name.replace("-", "_"), file_pattern)

        return re.compile(pattern)

    def _get_description(self) -> str:
        modality = self._modality
        if not modality.endswith("w"):
            modality = modality.upper()

        description = f"{modality} images registered to MNI152NLin2009cSym space using Clinica's '{self._pipeline_name}' pipeline"

        if not self.use_uncropped_image:
            description += (
                ", and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            )
        return description
