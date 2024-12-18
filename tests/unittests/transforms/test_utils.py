import torch
import torchio as tio

from clinicadl.transforms.utils import get_tio_image


def test_get_tio_image():
    image_tensor = torch.randn(1, 3, 4, 5)
    mask_1 = torch.ones(1, 3, 4, 5)
    mask_2 = torch.zeros(1, 3, 4, 5)
    label = torch.ones(1, 3, 4, 5)

    tio_image = get_tio_image(image_tensor, label, mask_1=mask_1, mask_2=mask_2)
    assert isinstance(tio_image.image, tio.ScalarImage)
    assert (tio_image.image.tensor == image_tensor).all()
    assert isinstance(tio_image.label, tio.LabelMap)
    assert (tio_image.label.tensor == label).all()
    assert isinstance(tio_image.mask_1, tio.LabelMap)
    assert (tio_image.mask_1.tensor == mask_1).all()
    assert isinstance(tio_image.mask_2, tio.LabelMap)
    assert (tio_image.mask_2.tensor == mask_2).all()

    tio_image = get_tio_image(image_tensor, label=None)
    assert tio_image.label is None
    tio_image = get_tio_image(image_tensor, label=1)
    assert tio_image.label == 1
