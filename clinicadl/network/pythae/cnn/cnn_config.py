from typing import Tuple

from pydantic import NonNegativeInt

from clinicadl.network.pythae.base import ModelConfig
from clinicadl.network.pythae.nn.networks.cnn import ImplementedCNN


class CNNConfig(ModelConfig):
    network: ImplementedCNN = ImplementedCNN.Conv5_FC3
    loss: ClassificationLoss = ClassificationLoss.CrossEntropyLoss
    input_size: Tuple[NonNegativeInt, ...]
    output_size: NonNegativeInt = 1
