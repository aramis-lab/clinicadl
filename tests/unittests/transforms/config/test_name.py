from clinicadl.transforms.config import *

MANDATORY_ARGS = {
    "masking_method": "mask",
    "remapping": {0: 1},
    "target_shape": 1,
    "target_multiple": 1,
    "cropping": 1,
    "padding": 1,
    "out_min": 0,
    "applied_labels": [0],
    "threshold": 0.5,
    "softmax": True,
    "transforms": [PadConfig(padding=0)],
    "keys": ["a", "b"],
    "output_key": "c",
}


def test_name():
    for name in ImplementedTransform:
        config = globals()[f"{name.value}Config"]
        c = config(**MANDATORY_ARGS)
        assert c.name == name.value
