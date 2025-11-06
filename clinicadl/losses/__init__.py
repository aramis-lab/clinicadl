"""To create a criterion to minimize during training."""

from .factory import get_loss_function_from_dict
from .types import Loss, LossOrConfig
