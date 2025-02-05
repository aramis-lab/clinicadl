from logging import getLogger

from pydantic import computed_field

from ..enum import PreprocessingMethod
from ..modalities import Flair
from .base import _LinearPreprocessing

logger = getLogger("clinicadl.data.datatype.preprocessing.flair")


class FlairLinear(_LinearPreprocessing, Flair):
    """
    Configuration class to handle Fluid-Attenuated Inversion Recovery (FLAIR) MRI images,
    preprocessed with Clinica's `flair-linear` pipeline.

    ..seealso::https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/FLAIR_Linear/

    Parameters
    ----------
    use_uncropped_image : bool (optional, default=False)
        whether to use the uncropped images returned by Clinica.
        - if `use_uncropped_image=True`: only the files that match the pattern
        `"flair_linear/sub-*_ses-*_space-MNI152NLin2009cSym_res-1x1x1_flair.nii*"`
        in the caps directory will be considered.
        - else: only the files that match the pattern
        `"flair_linear/sub-*_ses-*_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_flair.nii*"`
        in the caps directory will be considered.
    """

    @computed_field
    @property
    def preprocessing(self) -> str:
        """The preprocessing method."""
        return PreprocessingMethod.FLAIR_LINEAR.value
