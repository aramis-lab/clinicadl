from typing import Optional, Union

import torch
import torchio as tio


def get_tio_image(
    image: torch.Tensor,
    label: Optional[Union[float, int, torch.Tensor]],
    **masks: torch.Tensor,
) -> tio.Subject:
    """
    Creates a TorchIO Subject from the image, the label and possibly
    masks related to the image.

    Parameters
    ----------
    image : torch.Tensor
        the image, as a Pytorch tensor.
    label : Optional[Union[float, int, torch.Tensor]]
        the label related to the image. Can be None if no label.
    **masks : torch.Tensor
        any mask related to the image and useful to compute transforms.

    Returns
    -------
    tio.Subject
        the TorchIO subject with the image and the label, accessible via
        the attributes 'image' and 'label', as well as the masks, accessible
        via their names.
    """
    if isinstance(label, torch.Tensor):
        tio_image = tio.Subject(
            image=tio.ScalarImage(tensor=image), label=tio.LabelMap(tensor=label)
        )
    else:
        tio_image = tio.Subject(image=tio.ScalarImage(tensor=image), label=label)

    for name, mask in masks.items():
        tio_image.add_image(tio.LabelMap(tensor=mask), name)

    return tio_image
