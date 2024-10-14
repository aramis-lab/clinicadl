from typing import Any, Dict, List, Optional, Tuple, Union

from .enum import (
    ActFunction,
    NormLayer,
    PoolingLayer,
    UnpoolingLayer,
)

SingleLayerConvParameter = Union[int, Tuple[int, ...]]
ConvParameters = Union[SingleLayerConvParameter, List[SingleLayerConvParameter]]

PoolingType = Union[str, PoolingLayer]
SingleLayerPoolingParameters = Tuple[PoolingType, Dict[str, Any]]
PoolingParameters = Optional[
    Union[SingleLayerPoolingParameters, List[SingleLayerPoolingParameters]]
]

UnpoolingType = Union[str, UnpoolingLayer]
SingleLayerUnpoolingParameters = Tuple[UnpoolingType, Dict[str, Any]]
UnpoolingParameters = Optional[
    Union[SingleLayerUnpoolingParameters, List[SingleLayerUnpoolingParameters]]
]

NormalizationType = Union[str, NormLayer]
NormalizationParameters = Optional[
    Union[NormalizationType, Tuple[NormalizationType, Dict[str, Any]]]
]

ActivationType = Union[str, ActFunction]
ActivationParameters = Union[ActivationType, Tuple[ActivationType, Dict[str, Any]]]
