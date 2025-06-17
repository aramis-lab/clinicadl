from typing import Any, Union

from ..enum import PreprocessingMethod
from .base import Preprocessing
from .custom import Custom
from .dti import DWIDTI
from .flair import FlairLinear
from .pet import PETLinear
from .t1w import T1Linear


def get_preprocessing_config(
    name: Union[str, PreprocessingMethod], **kwargs: Any
) -> Preprocessing:
    """
    Factory function to get a Preprocessing object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, PreprocessingMethod]
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
    preprocessing = PreprocessingMethod(name)
    if preprocessing == PreprocessingMethod.T1_LINEAR:
        config = T1Linear
    elif preprocessing == PreprocessingMethod.FLAIR_LINEAR:
        config = FlairLinear
    elif preprocessing == PreprocessingMethod.PET_LINEAR:
        config = PETLinear
    elif preprocessing == PreprocessingMethod.DWI_DTI:
        config = DWIDTI
    elif preprocessing == PreprocessingMethod.CUSTOM:
        config = Custom

    return config(**kwargs)  # pylint: disable=possibly-used-before-assignment
