import pytest
from pydantic import ValidationError

from clinicadl.transforms.config import create_transform_config
from clinicadl.transforms.config.intensity_augmentations import RandomMotionConfig
from clinicadl.transforms.config.spatial_augmentations import RandomFlipConfig


def test_one_of():
    one_of = create_transform_config("OneOf")(
        transforms=[
            create_transform_config("RandomMotion")(degrees=1),
            create_transform_config("RandomFlip")(axes=1),
        ],
        probabilities=[10, 2],
    )
    assert [type(transform) for transform in one_of.transforms] == [
        RandomMotionConfig,
        RandomFlipConfig,
    ]
    assert one_of.transforms[0].degrees == 1
    assert one_of.transforms[1].axes == 1
    assert one_of.probabilities == [10, 2]

    one_of = create_transform_config("OneOf")(
        transforms=[
            create_transform_config("RandomMotion")(degrees=1),
            create_transform_config("RandomFlip")(axes=1),
        ],
    )
    assert one_of.probabilities == [1 / 2, 1 / 2]

    with pytest.raises(ValidationError):
        create_transform_config("OneOf")(
            transforms=[
                create_transform_config("RandomMotion")(degrees=1),
                create_transform_config("RandomFlip")(axes=1),
            ],
            probabilities=[10, -2],
        )

    with pytest.raises(ValidationError):
        create_transform_config("OneOf")(
            transforms=[
                create_transform_config("RandomMotion")(degrees=1),
                create_transform_config("RandomFlip")(axes=1),
            ],
            probabilities=[10],
        )
