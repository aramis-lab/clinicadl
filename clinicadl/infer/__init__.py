"""
To customize model inference, such as post-processing
or merging multiple neural network outputs.
"""

from .abstract import Inferer
from .patches_to_image import PatchesToImageInferer
from .simple import SimpleInferer
from .slices_to_image import SlicesToImageInferer
