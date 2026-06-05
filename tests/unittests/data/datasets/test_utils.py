import re
from copy import copy

import numpy as np
import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.data.datasets.utils import (
    MultimodalSamplerDataset,
    SamplerDataset,
)
from clinicadl.data.structures import DataPoint, Sample, Sample2D
from clinicadl.io.bids import T1Linear
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import CropConfig
from clinicadl.transforms.extraction import Patch, Slice
from clinicadl.utils.exceptions import DataFrameError

DF = pd.DataFrame(
    {
        "participant_id": ["sub-003", "sub-004", "sub-005", "sub-006"],
        "session_id": ["ses-M000", "ses-M001", "ses-M000", "ses-M000"],
        "age": [1.0, 2.0, 3.0, 4.0],
        "cat": ["A", "B", "C", "D"],
    }
)


def modify_age(x: pd.Series) -> pd.Series:
    return x * 10


def bad_cat_encoding(x):
    return 0 if x == "A" else 1


class Sampler(SamplerDataset):
    def __init__(self, transforms):
        super().__init__(transforms)
        self._df = copy(DF)

    def _get_data(self, participant, session):
        if participant == "sub-003":
            return DataPoint(
                participant_id=participant,
                session_id=session,
                image=tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3)),
                mask=tio.LabelMap(tensor=torch.randn(1, 3, 3, 3)),
                image_path="x",
                file_type=T1Linear(),
            )
        if participant == "sub-004":
            affine = np.eye(4) * 2
            affine_ = affine.copy()
            affine[:, 3] = [1, 1, 1, 1]
            return DataPoint(
                participant_id=participant,
                session_id=session,
                image=tio.ScalarImage(tensor=torch.randn(1, 4, 4, 4), affine=affine),
                mask=tio.LabelMap(tensor=torch.randn(1, 4, 4, 4), affine=affine_),
                image_path="x",
                file_type=T1Linear(),
            )
        if participant == "sub-005":
            return DataPoint(
                participant_id=participant,
                session_id=session,
                image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
                mask=tio.LabelMap(tensor=torch.randn(1, 1, 1, 1), affine=np.eye(4) * 2),
                image_path="x",
                file_type=T1Linear(),
            )
        if participant == "sub-006":
            return DataPoint(
                participant_id=participant,
                session_id=session,
                image=tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3)),
                mask=tio.LabelMap(tensor=torch.randn(1, 3, 3, 3)),
                image_path="x",
                file_type=T1Linear(),
            )


class MultimodalSampler(MultimodalSamplerDataset):
    def _get_images(self, participant, session):
        return Sampler._get_data(self, participant, session)


class TestSamplerDataset:
    def test_sort(self):
        dataset = Sampler(transforms=TransformsHandler())
        dataset._df = dataset.df.iloc[::-1]
        assert dataset[0].participant_id == "sub-006"
        dataset.sort()
        assert dataset[0].participant_id == "sub-003"

    def test_train_eval(self):
        dataset = Sampler(
            TransformsHandler(augmentations=[tio.Crop(cropping=(0, 1, 0, 1, 0, 1))])
        )
        assert not dataset.eval_mode
        assert dataset[0].image.spatial_shape == (2, 2, 2)
        dataset.eval()
        assert dataset.eval_mode
        assert dataset[0].image.spatial_shape == (3, 3, 3)
        dataset.train()
        assert not dataset.eval_mode

    def test_getitem(self):
        dataset = Sampler(
            transforms=TransformsHandler(
                extraction=Slice(squeeze=False),
                image_transforms=[
                    CropConfig(cropping=(0, 0, 0, 1, 0, 1)),
                ],
                sample_transforms=[
                    tio.RescaleIntensity(),
                ],
                augmentations=[tio.Pad(padding=(0, 0, 0, 1, 0, 0))],
            ),
        ).subset([("sub-003", "ses-M000"), ("sub-004", "ses-M001")])
        out_sample = dataset[2]
        assert isinstance(out_sample, Sample2D)
        assert out_sample.sample_position == 2
        assert out_sample.slice_direction == 0
        assert not out_sample.squeeze
        assert out_sample.participant_id == "sub-003"
        assert out_sample.session_id == "ses-M000"
        assert out_sample.spatial_shape == (1, 3, 2)

        dataset = Sampler(
            transforms=TransformsHandler(extraction=Patch(patch_size=2, overlap=0))
        )
        out_sample = dataset[7]
        assert isinstance(out_sample, Sample)
        assert out_sample.sample_position == (2, 2, 2)

        dataset = Sampler(transforms=TransformsHandler())
        out_sample = dataset[1]
        assert isinstance(out_sample, Sample)
        assert out_sample.participant_id == "sub-004"

    @pytest.mark.parametrize(
        "dataset,len_",
        [
            (
                Sampler(transforms=TransformsHandler(extraction=Slice())),
                12,
            ),
            (Sampler(transforms=TransformsHandler()), 4),
            (
                Sampler(
                    transforms=TransformsHandler(
                        extraction=Slice(), image_transforms=[tio.Crop(1)]
                    )
                ),
                None,
            ),
        ],
    )
    def test_len(self, dataset, len_):
        if len_:
            assert len(dataset) == len_
        else:
            with pytest.raises(
                RuntimeError,
                match=(
                    re.escape(
                        "An error occurred when reading the data of (sub-005, ses-M000) "
                        "to count the number of samples (see above)."
                    )
                ),
            ):
                len(dataset)

    def test_get_sample_info(self):
        dataset = Sampler(
            transforms=TransformsHandler(extraction=Patch(patch_size=2, overlap=0))
        )
        assert dataset.get_sample_info(8, "age") == 2.0
        assert dataset.get_sample_info(7, "age") == 1.0
        with pytest.raises(
            KeyError,
            match=re.escape(
                "No column named 'abc' in the metadata DataFrame. Present columns are: ['participant_id', 'session_id', 'age', 'cat', 'n_samples']"
            ),
        ):
            dataset.get_sample_info(0, "abc")
        with pytest.raises(
            IndexError, match="Index must be a non-negative integer, got -1."
        ):
            dataset.get_sample_info(-1, "age")
        with pytest.raises(
            IndexError,
            match="Index out of range, there are only 25 samples in total in the dataset.",
        ):
            dataset.get_sample_info(25, "age")

    def test_get_indices_associated_to(self):
        dataset = Sampler(transforms=TransformsHandler(extraction=Slice()))
        assert dataset._get_indices_associated_to("sub-003", "ses-M000") == (0, 2)
        assert dataset._get_indices_associated_to("sub-004", "ses-M001") == (3, 6)


class TestMulitmodalSamplerDataset:
    def test_df(self, tmp_path):
        dataset = MultimodalSampler(
            data=DF,
            columns=None,
            transforms=TransformsHandler(),
        )
        assert dataset.df is not DF
        pd.testing.assert_frame_equal(dataset.df, DF)
        assert set(dataset[0].keys()) == {
            "image",
            "participant_id",
            "session_id",
            "mask",
            "image_path",
            "file_type",
            "sample_position",
            "sample_type",
        }

        dataset = MultimodalSampler(
            data=DF,
            columns=["cat", "age"],
            transforms=TransformsHandler(extraction=Slice()),
        )
        assert dataset[3]["age"] == 2.0
        assert dataset[3]["cat"] == "B"

        DF.to_csv(tmp_path / "test.tsv", sep="\t", index=False)
        dataset = MultimodalSampler(
            data=tmp_path / "test.tsv",
            columns={"cat": None, "age": modify_age},
            transforms=TransformsHandler(extraction=Patch(patch_size=2, overlap=0)),
        )
        assert dataset[8]["age"] == 20.0
        assert dataset[8]["cat"] == "B"

    def test_checks(self):
        df = copy(DF)
        df["sample_type"] = None
        with pytest.raises(
            ValueError,
            match=r"A column cannot be named 'sample_type'. \('image', .*\) are protected names.",
        ):
            MultimodalSampler(
                data=df,
                columns=["sample_type"],
                transforms=TransformsHandler(),
            )

        with pytest.raises(
            KeyError,
            match=r"No column named 'abc' .*",
        ):
            MultimodalSampler(
                data=df,
                columns=["abc"],
                transforms=TransformsHandler(),
            )

        with pytest.raises(
            KeyError,
            match=r"No column named 'abc' .*",
        ):
            MultimodalSampler(
                data=df,
                columns=["abc"],
                transforms=TransformsHandler(),
            )

        with pytest.raises(
            RuntimeError,
            match="Unable to process the column 'cat' with the function you passed. Make sure that this function takes as input a Pandas Series, and returns a Pandas Series.",
        ):
            MultimodalSampler(
                data=df,
                columns={"cat": bad_cat_encoding},
                transforms=TransformsHandler(),
            )
