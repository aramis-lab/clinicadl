import re

import pytest

from clinicadl.io import BidsFileType, Tensor


class TestTensor:
    def test_init(self):
        tensor = Tensor(conversion_name="abc")
        assert tensor.extension == re.compile(".pt")
        assert tensor.suffix == re.compile("tensors")
        assert tensor.datatype == re.compile("tensors")
        assert tensor.without_entities is None
        assert tensor.description == "Outputs of the tensor conversion 'abc'."
        assert tensor.with_entities == {"conv": re.compile("abc")}

    def test_init_with_entities(self):
        tensor = Tensor(conversion_name="abc", entities={"trc": r"18FD.*"})
        assert tensor.with_entities == {
            "conv": re.compile("abc"),
            "trc": re.compile(r"18FD.*"),
        }

    @pytest.mark.parametrize(
        "sources,expected",
        [
            (
                (
                    BidsFileType(
                        datatype="",
                        suffix="",
                        with_entities={
                            "trc": "18FDG",
                            "res": "1x1x1",
                            "space": r"MNI.*",
                        },
                    ),
                    BidsFileType(
                        datatype="",
                        suffix="",
                        with_entities={
                            "trc": "18FDG",
                            "res": "2x2x2",
                            "space": "MNI",
                        },
                    ),
                ),
                {"trc": re.compile("18FDG")},
            ),
            (
                (
                    BidsFileType(
                        datatype="",
                        suffix="",
                        with_entities={
                            "res": "1x1x1",
                        },
                    ),
                    BidsFileType(
                        datatype="",
                        suffix="",
                        with_entities={
                            "res": "2x2x2",
                        },
                    ),
                ),
                {},
            ),
        ],
    )
    def test_from_source_file_types(self, sources, expected):
        tensor = Tensor.from_source_file_types(
            conversion_name="abc",
            file_types=sources,
        )
        expected["conv"] = re.compile("abc")
        assert tensor.with_entities == expected
