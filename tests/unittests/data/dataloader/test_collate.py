from copy import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.data.dataloader import (
    MergeBatchesCollate,
    ToBatchCollate,
    ToBatchesCollate,
)
from clinicadl.data.dataloader.collate.factory import get_collate_from_dict
from clinicadl.data.datatypes import T1Linear
from clinicadl.data.structures import Sample, Sample2D
from clinicadl.utils.json import read_json

SAMPLE_1 = Sample(
    image=tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3), affine=np.eye(4)),
    label=tio.LabelMap(tensor=torch.randn(1, 3, 3, 3), affine=np.eye(4)),
    participant=str(1),
    session=str(1),
    datatype=T1Linear(),
    image_path=Path("abc"),
    np_field=np.array([1, 2]),
    torch_field=torch.tensor([1, 2]),
    other_field=True,
)
SAMPLE_1_BIS = Sample(
    image=tio.ScalarImage(tensor=torch.randn(2, 3, 3, 3), affine=np.eye(4)),
    label=tio.LabelMap(tensor=torch.randn(3, 3, 3, 3), affine=np.eye(4)),
    participant=str(1),
    session=str(1),
    datatype=T1Linear(use_uncropped_image=True),
    image_path=Path("bcd"),
    np_field=np.array([2, 3]),
    torch_field=torch.tensor([2, 3]),
)
SAMPLE_2 = copy(SAMPLE_1)
SAMPLE_2["participant"] = str(2)
SAMPLE_2["session"] = str(2)


def test_to_batch():
    collate = ToBatchCollate()
    batch = collate([SAMPLE_1, SAMPLE_2])
    assert len(batch) == 2
    assert batch[0].participant == "1"
    assert batch[0].session == "1"


def test_to_batches():
    collate = ToBatchesCollate()
    batch = collate([(SAMPLE_1, SAMPLE_2), (SAMPLE_1, SAMPLE_2), (SAMPLE_1, SAMPLE_2)])
    assert len(batch[0]) == 3
    assert len(batch[1]) == 3
    assert batch[0][0].participant == "1"
    assert batch[1][0].participant == "2"


def test_merge_batches():
    collate = MergeBatchesCollate()
    batch = collate([(SAMPLE_1, SAMPLE_1_BIS), (SAMPLE_1, SAMPLE_1_BIS)])
    assert len(batch) == 2
    assert batch[0].image.shape == (3, 3, 3, 3)
    assert batch[0].label.shape == (4, 3, 3, 3)
    assert batch[0].participant == "1"
    assert batch[0].session == "1"
    assert batch[0].image_path == (Path("abc"), Path("bcd"), Path("bcd"))
    assert batch[0].datatype == (
        T1Linear(),
        T1Linear(use_uncropped_image=True),
        T1Linear(use_uncropped_image=True),
    )
    np.testing.assert_allclose(batch[0].np_field, np.array([[1, 2], [2, 3]]))
    torch.testing.assert_close(batch[0].torch_field, torch.tensor([[1, 2], [2, 3]]))
    assert batch[0].other_field

    ###
    collate = MergeBatchesCollate(ignore=["other_field", "participant"])
    assert batch[0].participant == "1"
    assert "other_field" not in batch

    ###
    sample = copy(SAMPLE_1_BIS)
    sample["label"] = None
    batch = collate([(SAMPLE_1, sample), (SAMPLE_1, sample)])
    assert batch[0].label.shape == (1, 3, 3, 3)

    ###
    sample["np_field"] = [0, 1, 2]
    with pytest.raises(
        TypeError,
        match="MergeBatchesCollate can only merge torchio.Image, numpy.ndarray, or torch.Tensor. For 'np_field', got:.*",
    ):
        collate([(SAMPLE_1, sample), (SAMPLE_1, sample)])

    ###
    collate = MergeBatchesCollate(ignore=["np_field"])
    sample.update({"sample_position": 0})
    sample.pop("sample_type")
    sample["image"] = tio.ScalarImage(tensor=torch.randn(2, 1, 3, 3), affine=np.eye(4))
    sample_2d = Sample2D(**sample, squeeze=True, slice_direction=0)
    with pytest.raises(
        TypeError, match="Cannot merge samples of different types. Got.*"
    ):
        collate([(SAMPLE_1, sample_2d), (SAMPLE_1, sample_2d)])
    batch = collate([(sample_2d, sample_2d), (sample_2d, sample_2d)])
    assert isinstance(batch[0], Sample2D)

    ###
    sample = copy(SAMPLE_1_BIS)
    sample["image"] = tio.ScalarImage(
        tensor=torch.randn(2, 3, 3, 3), affine=np.diag([1.2, 1.1, 1, 1])
    )
    with pytest.raises(
        RuntimeError, match="Trying to concatenate images with different voxel spacing!"
    ):
        collate([(SAMPLE_1, sample), (SAMPLE_1, sample)])

    with pytest.raises(RuntimeError, match="Got different values for 'participant':.*"):
        collate([(SAMPLE_1, SAMPLE_2), (SAMPLE_1, SAMPLE_2), (SAMPLE_1, SAMPLE_2)])


@pytest.mark.parametrize(
    "collate",
    [
        ToBatchCollate,
        ToBatchesCollate,
        MergeBatchesCollate,
    ],
)
def test_get_collate_from_dict(collate, tmp_path):
    c = collate()
    c.to_json(tmp_path / "config.json")
    dict_ = read_json(tmp_path / "config.json")
    c = get_collate_from_dict(dict_)
    assert isinstance(c, collate)

    if collate is MergeBatchesCollate:
        c = MergeBatchesCollate(ignore=["abc"])
        assert get_collate_from_dict(c.to_dict()).config.ignore == ["abc"]
