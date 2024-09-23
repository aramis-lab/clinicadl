from collections import OrderedDict

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def network():
    network = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 1, kernel_size=3)),
                ("dense1", nn.Linear(10, 10)),
            ]
        )
    )
    network.add_module(
        "final",
        nn.Sequential(
            OrderedDict([("dense2", nn.Linear(10, 5)), ("dense3", nn.Linear(5, 3))])
        ),
    )
    return network


def test_get_params_in_groups(network):
    import torch

    from clinicadl.optim.optimizer.utils import get_params_in_groups

    iterator, list_layers = get_params_in_groups(network, "dense1")
    assert next(iter(iterator)).shape == torch.Size((10, 10))
    assert next(iter(iterator)).shape == torch.Size((10,))
    assert sorted(list_layers) == sorted(["dense1.weight", "dense1.bias"])

    iterator, list_layers = get_params_in_groups(network, "dense1.weight")
    assert next(iter(iterator)).shape == torch.Size((10, 10))
    assert sum(1 for _ in iterator) == 0
    assert sorted(list_layers) == sorted(["dense1.weight"])

    iterator, list_layers = get_params_in_groups(network, "final.dense3")
    assert next(iter(iterator)).shape == torch.Size((3, 5))
    assert next(iter(iterator)).shape == torch.Size((3,))
    assert sorted(list_layers) == sorted(["final.dense3.weight", "final.dense3.bias"])

    iterator, list_layers = get_params_in_groups(network, "final")
    assert sum(1 for _ in iterator) == 4
    assert sorted(list_layers) == sorted(
        [
            "final.dense2.weight",
            "final.dense2.bias",
            "final.dense3.weight",
            "final.dense3.bias",
        ]
    )

    iterator, list_layers = get_params_in_groups(network, ["dense1.weight", "final"])
    assert sum(1 for _ in iterator) == 5
    assert sorted(list_layers) == sorted(
        [
            "dense1.weight",
            "final.dense2.weight",
            "final.dense2.bias",
            "final.dense3.weight",
            "final.dense3.bias",
        ]
    )

    # chrck with numbers
    network_bis = nn.Sequential(nn.Linear(10, 2), nn.Linear(2, 1))
    iterator, list_layers = get_params_in_groups(network_bis, "0.bias")
    assert next(iter(iterator)).shape == torch.Size((2,))
    assert sorted(list_layers) == sorted(["0.bias"])


def test_find_params_not_in_group(network):
    import torch

    from clinicadl.optim.optimizer.utils import get_params_not_in_groups

    iterator, list_layers = get_params_not_in_groups(
        network,
        [
            "final",
            "conv1.weight",
        ],
    )
    assert next(iter(iterator)).shape == torch.Size((1,))
    assert next(iter(iterator)).shape == torch.Size((10, 10))
    assert sum(1 for _ in iterator) == 1
    assert sorted(list_layers) == sorted(
        [
            "conv1.bias",
            "dense1.weight",
            "dense1.bias",
        ]
    )

    iterator, list_layers = get_params_not_in_groups(network, "final")
    assert sum(1 for _ in iterator) == 4
    assert sorted(list_layers) == sorted(
        [
            "conv1.weight",
            "conv1.bias",
            "dense1.weight",
            "dense1.bias",
        ]
    )
