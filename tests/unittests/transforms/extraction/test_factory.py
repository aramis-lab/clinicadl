import pytest

from clinicadl.transforms.extraction import (
    Image,
    Patch,
    Slice,
    get_extraction_from_dict,
)


@pytest.mark.parametrize(
    "extraction,params",
    [
        (Image, {}),
        (Slice, {"slice_direction": 2, "slices": [3, 4]}),
        (Patch, {"patch_size": (3, 2, 3), "overlap": (0.2, 0.2, 0.2)}),
    ],
)
def test_get_extraction_from_dict(extraction, params):
    extractor = extraction(**params)
    dict_ = extractor.to_dict()
    new_exctractor = get_extraction_from_dict(dict_)
    assert isinstance(new_exctractor, extraction)
    for param in params:
        assert getattr(new_exctractor.config, param) == params[param]
