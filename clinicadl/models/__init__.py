"""
To define **models**, which encompass **neural networks**, **loss functions**, **optimizers**,
and the logic to use them during **training** and **evaluation**.
"""

from .base import ClinicaDLModel
from .reconstruction import ReconstructionModel
from .supervised import SupervisedModel
