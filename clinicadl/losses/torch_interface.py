from enum import Enum


class ImplementedLosses(str, Enum):
    Conv5_FC3 = "Conv5_FC3"
    Conv4_FC3 = "Conv4_FC3"
    Stride_Conv5_FC3 = "Stride_Conv5_FC3"
    RESNET = "resnet18"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented networks are: "
            + ", ".join([repr(m.value) for m in cls])
        )
