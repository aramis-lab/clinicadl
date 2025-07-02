from logging import getLogger

from pydantic import computed_field

from ..enum import PreprocessingMethod
from ..modalities import T1w
from .base import _LinearPreprocessing

logger = getLogger("clinicadl.data.datatypes.preprocessing.t1")


class T1Linear(_LinearPreprocessing, T1w):
    """
    Configuration class to handle T1-weighted MRI images,
    preprocessed with `Clinica t1-linear <https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/T1_Linear/>`_
    pipeline.

    Parameters
    ----------
    use_uncropped_image : bool, default=False
        Whether to use the uncropped images returned by ``Clinica``:\n
        - if ``use_uncropped_image=True``: only the files that match the pattern
          ``t1_linear/sub-*_ses-*_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii*``
          in the :term:`CAPS` structure will be considered.
        - else: only the files that match the pattern
          ``t1_linear/sub-*_ses-*_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii*``
          in the :term:`CAPS` structure will be considered.
    """

    @computed_field
    @property
    def name(self) -> str:
        """The preprocessing method."""
        return PreprocessingMethod.T1_LINEAR.value
