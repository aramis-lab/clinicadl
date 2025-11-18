from ..modalities import Flair
from .base import _LinearPreprocessing


class FlairLinear(_LinearPreprocessing, Flair):
    """
    :py:class:`DataType <clinicadl.data.datatypes.DataType>` to handle Fluid-Attenuated Inversion Recovery (FLAIR) MRI images
    preprocessed with `Clinica flair-linear <https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/FLAIR_Linear/>`_
    pipeline.

    Parameters
    ----------
    use_uncropped_image : bool, default=False
        Whether to use the uncropped images returned by ``Clinica``:\n
        - if ``use_uncropped_image=True``: only the files that match the pattern
          ``flair_linear/sub-*_ses-*_space-MNI152NLin2009cSym_res-1x1x1_FLAIR.nii*``
          in the :term:`CAPS` structure will be considered.
        - else: only the files that match the pattern
          ``flair_linear/sub-*_ses-*_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_FLAIR.nii*``
          in the :term:`CAPS` structure will be considered.
    """

    @property
    def _pipeline_name(self) -> str:
        return "flair-linear"
