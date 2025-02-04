from typing import Any, Union

from .base import Preprocessing
from .custom import CustomPreprocessing
from .dti import DWIDTI
from .enum import PreprocessingMethod
from .flair import FlairLinear
from .pet import PETLinear
from .t1 import T1Linear


def get_preprocessing_config(
    preprocessing: Union[str, PreprocessingMethod], **kwargs: Any
) -> Preprocessing:
    """
    Factory function to get a Preprocessing object from its name
    and parameters.

    Parameters
    ----------
    preprocessing : Union[str, PreprocessingMethod]
        the name of the preprocessing. Check our documentation to know
        supported neuroimaging preprocessings.
    **kwargs : Any
        any preprocessing parameter. Check our documentation on preprocessings to
        know these parameters.

    Returns
    -------
    Preprocessing
        the Preprocessing object.
    """
    preprocessing = PreprocessingMethod(preprocessing)
    if preprocessing == PreprocessingMethod.T1_LINEAR:
        config = T1Linear
    elif preprocessing == PreprocessingMethod.FLAIR_LINEAR:
        config = FlairLinear
    elif preprocessing == PreprocessingMethod.PET_LINEAR:
        config = PETLinear
    elif preprocessing == PreprocessingMethod.DWI_DTI:
        config = DWIDTI
    elif preprocessing == PreprocessingMethod.CUSTOM:
        config = CustomPreprocessing

    return config(**kwargs)  # pylint: disable=possibly-used-before-assignment
