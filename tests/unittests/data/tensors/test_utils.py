from pathlib import Path

from clinicadl.data.tensors.utils import path_to_tensors


def test_path_to_tensors():
    assert path_to_tensors(
        path=Path("abc") / "abc.nii.gz.etc",
        tensors_location=Path("tensors") / "conversion",
    ) == (Path("abc") / "tensors" / "conversion" / "abc.pt")
