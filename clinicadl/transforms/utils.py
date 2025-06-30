import torchio as tio

from clinicadl.data.structures import DataPoint
from clinicadl.utils.enum import SliceDirection


class Squeeze(tio.Transform):
    """
    To squeeze a spatial dimension to get 2D slices.

    Parameters
    ----------
    direction : int
        Which spatial dimension to squeeze.
    """

    def __init__(
        self,
        direction: SliceDirection,
    ):
        super().__init__()
        self.direction = direction
        self.direction = ["direction"]

    def apply_transform(self, datapoint: DataPoint) -> DataPoint:  # pylint: disable=arguments-renamed
        """
        Apply the transform to the datapoint.
        """
        for image in datapoint.get_images(intensity_only=True):
            image.set_data(image.tensor.squeeze(self.direction + 1))
        return datapoint
