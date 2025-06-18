"""Config classes for optimizers supported natively in ``ClinicaDL``. Based
on :torch:`PyTorch optimizers <optim.html#algorithms>`."""

from .base import ImplementedOptimizer, OptimizerConfig
from .configs import *
from .factory import get_optimizer_config
