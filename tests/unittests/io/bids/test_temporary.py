import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torchio as tio

from clinicadl.data.datasets import BidsDataset
from clinicadl.io.bids import Bids, BidsFileType, TemporaryBidsDir
from clinicadl.utils.exceptions import DataFrameError


@pytest.fixture
def description():
    return {"Name": "Toy dataset", "BIDSVersion": "1.10.0", "DatasetType": "raw"}


@pytest.fixture
def metadata():
    return pd.DataFrame(
        {
            "participant_id": ["sub-001", "sub-001", "sub-002"],
            "session_id": ["ses-M000", "ses-M006", "ses-M000"],
            "age": [55.0, 55.5, 62.0],
        }
    )


def test_navigation_without_metadata(description):
    temporary = TemporaryBidsDir(description, metadata=None)
    assert isinstance(temporary, tempfile.TemporaryDirectory)
    with temporary as directory:
        assert isinstance(directory, Path)
        assert (
            json.loads((directory / "dataset_description.json").read_text())
            == description
        )
        assert not (directory / "metadata.tsv").exists()
        (directory / "sub-01" / "ses-M000").mkdir(parents=True)
        assert Bids(directory).get_all_participants_sessions() == {
            ("sub-01", "ses-M000")
        }
    assert not directory.exists()


def test_metadata_without_images(description, metadata):
    original = metadata.copy(deep=True)
    with TemporaryBidsDir(description, metadata) as directory:
        pd.testing.assert_frame_equal(
            pd.read_csv(directory / "metadata.tsv", sep="\t"), metadata
        )
        assert sorted(p.name for p in directory.iterdir()) == [
            "dataset_description.json",
            "metadata.tsv",
        ]
    pd.testing.assert_frame_equal(metadata, original)
    assert not directory.exists()


@pytest.mark.parametrize("shape", [(1, 8, 8, 8), (1, 3, 4, 5), (2, 3, 4, 5)])
def test_images_for_every_pair_and_data_type(description, metadata, shape):
    endings = {"anat": "res-1x1x1_T1w", "pet": "trc-18FFDG_pet"}
    with TemporaryBidsDir(
        description, metadata, endings, image_shape=shape
    ) as directory:
        assert len(list(directory.rglob("*.nii.gz"))) == len(metadata) * 2
        for participant, session in metadata[
            ["participant_id", "session_id"]
        ].itertuples(index=False, name=None):
            for data_type, suffix in endings.items():
                path = (
                    directory
                    / participant
                    / session
                    / data_type
                    / f"{participant}_{session}_{suffix}.nii.gz"
                )
                image = tio.ScalarImage(path)
                assert image.shape == shape
                assert image.data.count_nonzero() == 0
                np.testing.assert_array_equal(image.affine, np.eye(4))
                if shape[0] == 1:
                    assert nib.load(path).shape == shape[1:]
        assert Bids(directory).get_all_participants_sessions() == set(
            metadata[["participant_id", "session_id"]].itertuples(
                index=False, name=None
            )
        )
    assert not directory.exists()


def test_dataset_loading(description, metadata):
    with TemporaryBidsDir(description, metadata, {"anat": "T1w"}) as directory:
        dataset = BidsDataset(
            bids=directory,
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
            data=directory / "metadata.tsv",
        )
        assert len(dataset) == 3
        for index, row in metadata.iterrows():
            sample = dataset[index]
            assert (sample.participant_id, sample.session_id) == (
                row.participant_id,
                row.session_id,
            )
            assert sample.spatial_shape == (8, 8, 8)


def test_cleanup_on_exception(description, tmp_path):
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("keep")
    with pytest.raises(RuntimeError, match="example failed"):
        with TemporaryBidsDir(description) as directory:
            raise RuntimeError("example failed")
    assert not directory.exists()
    assert sentinel.read_text() == "keep"


def test_explicit_cleanup(description):
    temporary = TemporaryBidsDir(description)
    directory = Path(temporary.name)
    assert (directory / "dataset_description.json").is_file()
    temporary.cleanup()
    assert not directory.exists()


def test_cleanup_on_image_save_failure(description, metadata, monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def fail_save(self, path):
        raise OSError("save failed")

    monkeypatch.setattr(tio.ScalarImage, "save", fail_save)
    with pytest.raises(OSError, match="save failed"):
        TemporaryBidsDir(description, metadata, {"anat": "T1w"})
    assert list(tmp_path.iterdir()) == []


def test_images_require_metadata(description):
    with pytest.raises(ValueError, match="requires metadata"):
        TemporaryBidsDir(description, with_images={"anat": "T1w"})


@pytest.mark.parametrize("column", ["participant_id", "session_id"])
@pytest.mark.parametrize("value", [None, 123, "wrong-prefix"])
def test_invalid_metadata(description, metadata, column, value):
    metadata[column] = metadata[column].astype(object)
    metadata.loc[0, column] = value
    with pytest.raises(DataFrameError):
        TemporaryBidsDir(description, metadata, {"anat": "T1w"})


@pytest.mark.parametrize(
    "column,value",
    [
        ("participant_id", "sub-../../outside"),
        ("session_id", "ses-../../outside"),
        ("participant_id", "sub-..\\outside"),
        ("session_id", "ses-C:outside"),
    ],
)
def test_identifiers_cannot_escape_directory(
    description, metadata, column, value, monkeypatch, tmp_path
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    metadata.loc[0, column] = value
    with pytest.raises(ValueError, match="Invalid BIDS"):
        TemporaryBidsDir(description, metadata, {"anat": "T1w"})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "images",
    [
        {"../anat": "T1w"},
        {"C:\\anat": "T1w"},
        {"anat": "../../outside"},
        {"anat": "..\\outside"},
        {"anat": "T1w.nii.gz"},
        {"anat": ""},
    ],
)
def test_invalid_image_paths(description, metadata, images):
    with pytest.raises(ValueError, match="Invalid BIDS"):
        TemporaryBidsDir(description, metadata, images)


@pytest.mark.parametrize(
    "shape", [(8, 8, 8), (1, 0, 8, 8), (1, -1, 8, 8), (1, 1.5, 8, 8), (True, 8, 8, 8)]
)
def test_invalid_image_shapes(description, metadata, shape):
    with pytest.raises(ValueError, match="four positive integers"):
        TemporaryBidsDir(description, metadata, {"anat": "T1w"}, image_shape=shape)
