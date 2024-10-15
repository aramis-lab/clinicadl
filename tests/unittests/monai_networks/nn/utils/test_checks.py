import pytest

from clinicadl.monai_networks.nn.utils.checks import (
    _check_conv_parameter,
    check_conv_args,
    check_mlp_args,
    check_norm_layer,
    check_pool_indices,
    ensure_list_of_tuples,
)


@pytest.mark.parametrize(
    "parameter,expected_output",
    [
        (5, (5, 5, 5)),
        ((5, 4, 4), (5, 4, 4)),
        ([5, 4], [(5, 5, 5), (4, 4, 4)]),
        ([5, (4, 3, 3)], [(5, 5, 5), (4, 3, 3)]),
        ((5, 5), None),
        ([5, 5, 5], None),
        ([5, (4, 4)], None),
        (5.0, None),
    ],
)
def test_check_conv_parameter(parameter, expected_output):
    if expected_output:
        assert (
            _check_conv_parameter(parameter, dim=3, n_layers=2, name="abc")
            == expected_output
        )
    else:
        with pytest.raises(ValueError):
            _check_conv_parameter(parameter, dim=3, n_layers=2, name="abc")


@pytest.mark.parametrize(
    "parameter,expected_output",
    [
        (5, [(5, 5, 5), (5, 5, 5)]),
        ((5, 4, 4), [(5, 4, 4), (5, 4, 4)]),
        ([5, 4], [(5, 5, 5), (4, 4, 4)]),
        ([5, (4, 3, 3)], [(5, 5, 5), (4, 3, 3)]),
    ],
)
def test_ensure_list_of_tuples(parameter, expected_output):
    assert (
        ensure_list_of_tuples(parameter, dim=3, n_layers=2, name="abc")
        == expected_output
    )


@pytest.mark.parametrize(
    "indices,n_layers,error",
    [
        ([0, 1, 2], 4, False),
        ([0, 1, 2], 3, False),
        ([-1, 1, 2], 3, False),
        ([0, 1, 2], 2, True),
        ([-2, 1, 2], 3, True),
    ],
)
def test_check_pool_indices(indices, n_layers, error):
    if error:
        with pytest.raises(ValueError):
            _ = check_pool_indices(indices, n_layers)
    else:
        check_pool_indices(indices, n_layers)


@pytest.mark.parametrize(
    "inputs,error",
    [
        (None, False),
        ("abc", True),
        ("batch", False),
        ("group", True),
        (("batch",), True),
        (("batch", 3), True),
        (("batch", {"eps": 0.1}), False),
        (("group", {"num_groups": 2}), False),
        (("group", {"num_groups": 2, "eps": 0.1}), False),
    ],
)
def test_check_norm_layer(inputs, error):
    if error:
        with pytest.raises(ValueError):
            _ = check_norm_layer(inputs)
    else:
        assert check_norm_layer(inputs) == inputs


@pytest.mark.parametrize(
    "conv_args,error",
    [(None, True), ({"kernel_size": 3}, True), ({"channels": [2]}, False)],
)
def test_check_conv_args(conv_args, error):
    if error:
        with pytest.raises(ValueError):
            check_conv_args(conv_args)
    else:
        check_conv_args(conv_args)


@pytest.mark.parametrize(
    "mlp_args,error",
    [({"act": "tanh"}, True), ({"hidden_channels": [2]}, False)],
)
def test_check_mlp_args(mlp_args, error):
    if error:
        with pytest.raises(ValueError):
            check_mlp_args(mlp_args)
    else:
        check_mlp_args(mlp_args)
