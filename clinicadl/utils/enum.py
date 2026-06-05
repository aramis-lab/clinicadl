from enum import Enum


class CaseInsensitiveEnum(str, Enum):
    """Case insensitive Enum object."""

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.lower()
        for member in cls:
            if member.lower() == value:
                return member
        return None


class BaseEnum(Enum):
    """Base Enum object that will print valid inputs if the value passed is not valid."""

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Valid ones are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class SliceDirection(int, Enum):
    """Possible directions for a slice."""

    SAGITTAL = 0
    CORONAL = 1
    AXIAL = 2


class TrainerStage(str, Enum):
    """Possible stages of the :py:class:`clinicadl.train.Trainer`."""

    TRAIN = "training"
    EVAL = "evaluation"
    PRED = "prediction"
    INTERRUPTED = "interrupted"


class TrainerCall(str, Enum):
    """Public methods of :py:class:`clinicadl.train.Trainer`."""

    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"
