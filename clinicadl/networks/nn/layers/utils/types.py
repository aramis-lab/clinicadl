from typing import Any, Union

from .enum import (
    ActFunction,
    ConvNormLayer,
    NormLayer,
    PoolingLayer,
    UnpoolingLayer,
)

SingleLayerConvParameter = Union[int, tuple[int, ...]]
ConvParameters = Union[SingleLayerConvParameter, list[SingleLayerConvParameter]]

SingleLayerPoolingParameters = tuple[PoolingLayer, dict[str, Any]]
PoolingParameters = Union[
    SingleLayerPoolingParameters, list[SingleLayerPoolingParameters]
]

SingleLayerUnpoolingParameters = tuple[UnpoolingLayer, dict[str, Any]]
UnpoolingParameters = Union[
    SingleLayerUnpoolingParameters, list[SingleLayerUnpoolingParameters]
]

NormalizationParameters = Union[NormLayer, tuple[NormLayer, dict[str, Any]]]

ConvNormalizationParameters = Union[ConvNormLayer, tuple[ConvNormLayer, dict[str, Any]]]

ActivationParameters = Union[ActFunction, tuple[ActFunction, dict[str, Any]]]
