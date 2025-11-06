from clinicadl.transforms.config import *


def test_name():
    for name in ImplementedTransform:
        config = globals()[f"{name.value}Config"]
    c = config()
    assert c.name == name.value
