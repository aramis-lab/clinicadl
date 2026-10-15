from importlib.metadata import version

__all__ = ["__version__"]

__version__ = version("clinicadl")

import torch

content = torch.load("path", weights_only=False)
