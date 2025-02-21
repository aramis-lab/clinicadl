from typing import Optional

import torch
import torchio as tio


class NanRemoval(tio.IntensityTransform):
    """
    Replaces NaN, positive infinity, and negative infinity values.

    See also :py:func:`torch.nan_to_num`.

    Parameters
    ----------
    nan : float (optional, default=0.0)
        the value to replace NaN with.
    posinf : Optional[float] (optional, default=None)
        the value to replace positive infinity values with. If None,
        positive infinity values are replaced with the greatest finite
        value representable by image’s dtype.
    neginf : Optional[float] (optional, default=None)
        the value to replace negative infinity values with. If None,
        negative infinity values are replaced with the lowest finite
        value representable by image’s dtype.
    """

    def __init__(
        self,
        nan: float = 0.0,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nan, self.posinf, self.neginf = nan, posinf, neginf
        self.args_names = ["nan", "posinf", "neginf"]

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for image in self.get_images(subject):
            assert isinstance(image, tio.ScalarImage)
            self._apply_nan_removal(image)
        return subject

    def _apply_nan_removal(self, image: tio.ScalarImage) -> None:
        image.set_data(self._nan_removal(image.data))

    def _nan_removal(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.nan_to_num(self.nan, self.posinf, self.neginf)
