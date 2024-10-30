from enum import Enum


class ImplementedOptimizer(str, Enum):
    """Implemented optimizers in ClinicaDL."""

    ADADELTA = "Adadelta"
    ADAGRAD = "Adagrad"
    ADAM = "Adam"
    RMS_PROP = "RMSprop"
    SGD = "SGD"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented optimizers are: "
            + ", ".join([repr(m.value) for m in cls])
        )
