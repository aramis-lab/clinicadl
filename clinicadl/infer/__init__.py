"""
For customizing the inference stage, such as performing post-processing or
combining outputs from multiple neural networks.
"""

from .abstract import Inferer
from .patches_to_image import PatchesToImageInferer
from .simple import SimpleInferer
from .slices_to_image import SlicesToImageInferer
