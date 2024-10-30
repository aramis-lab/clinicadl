from clinicadl.utils.enum import CaseInsensitiveEnum


class UnpoolingLayer(CaseInsensitiveEnum):
    """Supported unpooling layers in ClinicaDL."""

    CONV_TRANS = "convtranspose"
    UPSAMPLE = "upsample"


class ActFunction(CaseInsensitiveEnum):
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
    MISH = "mish"


class PoolingLayer(CaseInsensitiveEnum):
    """Supported pooling layers in ClinicaDL."""

    MAX = "max"
    AVG = "avg"
    ADAPT_AVG = "adaptiveavg"
    ADAPT_MAX = "adaptivemax"


class NormLayer(CaseInsensitiveEnum):
    """Supported normalization layers in ClinicaDL."""

    GROUP = "group"
    LAYER = "layer"
    SYNCBATCH = "syncbatch"
    BATCH = "batch"
    INSTANCE = "instance"


class ConvNormLayer(CaseInsensitiveEnum):
    """Supported normalization layers with convolutions in ClinicaDL."""

    GROUP = "group"
    SYNCBATCH = "syncbatch"
    BATCH = "batch"
    INSTANCE = "instance"


class UnpoolingMode(CaseInsensitiveEnum):
    """Supported unpooling mode for AutoEncoders in ClinicaDL."""

    NEAREST = "nearest"
    LINEAR = "linear"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    TRILINEAR = "trilinear"
    CONV_TRANS = "convtranspose"
