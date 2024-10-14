from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..layers.utils import (
    ConvParameters,
    NormalizationParameters,
    NormLayer,
    PoolingLayer,
)

__all__ = [
    "ensure_list_of_tuples",
    "check_norm_layer",
    "check_conv_args",
    "check_mlp_args",
    "check_pool_indices",
]


def ensure_list_of_tuples(
    parameter: ConvParameters, dim: int, n_layers: int, name: str
) -> List[Tuple[int, ...]]:
    """
    Checks spatial parameters (e.g. kernel_size) and returns a list of tuples.
    Each element of the list corresponds to the parameters of one layer, and
    each element of the tuple corresponds to the parameters for one dimension.
    """
    parameter = _check_conv_parameter(parameter, dim, n_layers, name)
    if isinstance(parameter, tuple):
        return [parameter] * n_layers if n_layers > 0 else [parameter]
    else:
        return parameter


def check_norm_layer(
    norm: Optional[NormalizationParameters],
) -> Optional[NormalizationParameters]:
    """
    Checks that the argument for normalization layers has the right format (i.e.
    `norm_type` or (`norm_type`, `norm_layer_parameters`)) and checks potential
    mandatory arguments in `norm_layer_parameters`.
    """
    if norm is None:
        return norm

    if not isinstance(norm, str) and not isinstance(norm, PoolingLayer):
        if (
            not isinstance(norm, tuple)
            or len(norm) != 2
            or not isinstance(norm[1], dict)
        ):
            raise ValueError(
                "norm must be either the name of the normalization layer or a double with first the name and then the "
                f"arguments of the layer in a dict. Got {norm}"
            )
        norm_mode = NormLayer(norm[0])
        args = norm[1]
    else:
        norm_mode = NormLayer(norm)
        args = {}
    if norm_mode == NormLayer.GROUP and "num_groups" not in args:
        raise ValueError(
            f"num_groups is a mandatory argument for GroupNorm and must be passed in `norm`. Got `norm`={norm}"
        )

    return norm


def check_conv_args(conv_args: Dict[str, Any]) -> None:
    """
    Checks that `conv_args` is a dict with at least the mandatory argument `channels`.
    """
    if not isinstance(conv_args, dict):
        raise ValueError(
            f"conv_args must be a dict with the arguments for the convolutional part. Got: {conv_args}"
        )
    if "channels" not in conv_args:
        raise ValueError(
            "channels is a mandatory argument for the convolutional part and must therefore be "
            f"passed in conv_args. Got conv_args={conv_args}"
        )


def check_mlp_args(mlp_args: Optional[Dict[str, Any]]) -> None:
    """
    Checks that `mlp_args` is a dict with at least the mandatory argument `hidden_channels`.
    """
    if mlp_args is not None:
        if not isinstance(mlp_args, dict):
            raise ValueError(
                f"mlp_args must be a dict with the arguments for the MLP part. Got: {mlp_args}"
            )
        if "hidden_channels" not in mlp_args:
            raise ValueError(
                "hidden_channels is a mandatory argument for the MLP part and must therefore be "
                f"passed in mlp_args. Got mlp_args={mlp_args}"
            )


def check_pool_indices(
    pooling_indices: Optional[Sequence[int]], n_layers: int
) -> Sequence[int]:
    """
    Checks that the (un)pooling indices are consistent with the number of layers.
    """
    if pooling_indices is not None:
        for idx in pooling_indices:
            if idx > n_layers - 1:
                raise ValueError(
                    f"indices in (un)pooling_indices must be smaller than len(channels)-1, got (un)pooling_indices={pooling_indices} and len(channels)={n_layers}"
                )
        return pooling_indices
    else:
        return []


def _check_conv_parameter(
    parameter: ConvParameters, dim: int, n_layers: int, name: str
) -> Union[Tuple[int, ...], List[Tuple[int, ...]]]:
    """
    Checks spatial parameters (e.g. kernel_size).
    """
    if isinstance(parameter, int):
        return (parameter,) * dim
    elif isinstance(parameter, tuple):
        if len(parameter) != dim:
            raise ValueError(
                f"If a tuple is passed for {name}, its dimension must be {dim}. Got {parameter}"
            )
        return parameter
    elif isinstance(parameter, list):
        if len(parameter) != n_layers:
            raise ValueError(
                f"If a list is passed, {name} must contain as many elements as there are layers. "
                f"There are {n_layers} layers, but got {parameter}"
            )
        checked_params = []
        for param in parameter:
            checked_params.append(_check_conv_parameter(param, dim, n_layers, name))
        return checked_params
    else:
        raise ValueError(f"{name} must be an int, a tuple or a list. Got {name}")
