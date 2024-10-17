from typing import Any, Dict, List, Tuple, Union

from .enum import (
    ActFunction,
    ConvNormLayer,
    NormLayer,
    PoolingLayer,
    UnpoolingLayer,
)

SingleLayerConvParameter = Union[int, Tuple[int, ...]]
ConvParameters = Union[SingleLayerConvParameter, List[SingleLayerConvParameter]]

PoolingType = Union[str, PoolingLayer]
SingleLayerPoolingParameters = Tuple[PoolingType, Dict[str, Any]]
PoolingParameters = Union[
    SingleLayerPoolingParameters, List[SingleLayerPoolingParameters]
]

UnpoolingType = Union[str, UnpoolingLayer]
SingleLayerUnpoolingParameters = Tuple[UnpoolingType, Dict[str, Any]]
UnpoolingParameters = Union[
    SingleLayerUnpoolingParameters, List[SingleLayerUnpoolingParameters]
]

NormalizationType = Union[str, NormLayer]
NormalizationParameters = Union[
    NormalizationType, Tuple[NormalizationType, Dict[str, Any]]
]

ConvNormalizationType = Union[str, ConvNormLayer]
ConvNormalizationParameters = Union[
    ConvNormalizationType, Tuple[ConvNormalizationType, Dict[str, Any]]
]

ActivationType = Union[str, ActFunction]
ActivationParameters = Union[ActivationType, Tuple[ActivationType, Dict[str, Any]]]
