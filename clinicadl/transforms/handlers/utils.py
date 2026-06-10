from ..monai_wrapper import MonaiTransformWrapper
from ..types import Transform


def get_transform_name(transform: Transform) -> str:
    """
    Gets the name of the transform, even if it is wrapped
    in a :py:class:`clinicadl.transforms.monai_wrapper.MonaiTransformWrapper`.

    Parameters
    ----------
    transform : Transform
        The transform.

    Returns
    -------
    str
        Its name.
    """
    if isinstance(transform, MonaiTransformWrapper):
        transform = transform.transform
    return type(transform).__name__
