from enum import Enum


class CaseInsensitiveEnum(str, Enum):
    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.lower()
        for member in cls:
            if member.lower() == value:
                return member
        return None


class ImplementedActFunctions(CaseInsensitiveEnum):
    """Supported activation functions in ClinicaDL."""

    ELU = "elu"
    RELU = "relu"
    LEAKY_RELU = "leakyrelu"
    PRELU = "prelu"
    RELU6 = "relu6"
    SELU = "selu"
    CELU = "celu"
    GELU = "gelu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LOGSOFTMAX = "logsoftmax"
    SWISH = "swish"
    MEMSWISH = "memswish"
    MISH = "mish"
    GEGLU = "geglu"


class ImplementedNormLayers(CaseInsensitiveEnum):
    """Supported normalization layers in ClinicaDL."""

    GROUP = "group"
    LAYER = "layer"
    LOCAL_RESPONSE = "localresponse"
    SYNCBATCH = "syncbatch"
    INSTANCE_NVFUSER = "instance_nvfuser"
    BATCH = "batch"
    INSTANCE = "instance"
