import pytest

from clinicadl.infer import *
from clinicadl.infer.factory import ImplementedInferer, get_inferer_from_dict
from clinicadl.transforms.config import ActivationsConfig
from clinicadl.utils.json import read_json

MANDATORY_ARGS = {
    "PatchesToImageInferer": {"patch_size": 3},
    "SlicesToImageInferer": {"slice_direction": 0},
    "SimpleInferer": {},
}


@pytest.mark.parametrize(
    "inferer",
    [globals()[name.value] for name in ImplementedInferer],
)
def test_get_inferer_from_dict(inferer, tmp_path):
    obj = inferer(
        **MANDATORY_ARGS[inferer.__name__],
        postprocessing=[ActivationsConfig(sigmoid=True)],
    )
    obj.to_json(tmp_path / "config.json")
    dict_ = read_json(tmp_path / "config.json")
    obj = get_inferer_from_dict(dict_)
    assert isinstance(obj, inferer)

    if inferer is SimpleInferer:
        assert isinstance(
            obj.config.postprocessing.config.transforms.values[0].value,
            ActivationsConfig,
        )
