import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd
import torch
import torchio as tio

from clinicadl.utils.tsvtools import read_df


class TemporaryBidsDir(TemporaryDirectory):
    """Create a temporary toy BIDS directory for examples and tests.

    The directory is created at initialization and removed on context exit or
    :meth:`cleanup`, just like :class:`tempfile.TemporaryDirectory`. Entering the
    context returns a :class:`pathlib.Path`. Setup failures also remove it.

    Parameters
    ----------
    description : dict
        JSON-serializable dataset description, written to
        ``dataset_description.json``.
    metadata : pandas.DataFrame, optional
        Participant/session metadata, written to ``metadata.tsv``. Must contain
        ``participant_id`` and ``session_id`` columns. Without images, no
        participant/session directories are created automatically.
    with_images : dict[str, str], optional
        Map data types to filename endings, e.g. ``{"anat": "T1w"}`` or
        ``{"pet": "trc-18FFDG_pet"}``. Creates a zero-valued NIfTI image for
        every metadata row and data type. Requires metadata. Identifiers, data
        types and filename endings must use BIDS alphanumeric labels/entities.
    image_shape : tuple[int, int, int, int], default=(1, 8, 8, 8)
        Image tensor shape in TorchIO order: ``(channels, x, y, z)``. In particular,
        ``(1, 8, 8, 8)`` produces a scalar image with spatial shape ``(8, 8, 8)``.

    Notes
    -----
    These constant-valued images are for demonstrating APIs, not brain scans or
    training data. This helper does not validate a complete BIDS dataset.
    """

    def __init__(
        self,
        description: dict,
        metadata: Optional[pd.DataFrame] = None,
        with_images: Optional[dict[str, str]] = None,
        image_shape: tuple[int, int, int, int] = (1, 8, 8, 8),
    ):
        description_json = json.dumps(description)
        if metadata is not None:
            metadata = read_df(metadata)
        if with_images:
            if metadata is None:
                raise ValueError("Creating images requires metadata.")
            for column, prefix in (("participant_id", "sub"), ("session_id", "ses")):
                for value in metadata[column]:
                    self._check_component(value, rf"{prefix}-[a-zA-Z0-9]+", column)
            for data_type, suffix in with_images.items():
                self._check_component(data_type, r"[a-zA-Z0-9]+", "data type")
                self._check_component(
                    suffix,
                    r"(?:[a-zA-Z0-9]+-[a-zA-Z0-9]+_)*[a-zA-Z0-9]+",
                    "filename ending",
                )
            if len(image_shape) != 4 or any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0
                for size in image_shape
            ):
                raise ValueError("image_shape must contain four positive integers.")

        super().__init__()
        try:
            directory = Path(self.name)
            (directory / "dataset_description.json").write_text(
                description_json, encoding="utf-8"
            )
            if metadata is not None:
                metadata.to_csv(directory / "metadata.tsv", sep="\t", index=False)
            if with_images:
                image = tio.ScalarImage(tensor=torch.zeros(image_shape))
                for participant, session in metadata[
                    ["participant_id", "session_id"]
                ].itertuples(index=False, name=None):
                    for data_type, suffix in with_images.items():
                        image_directory = directory / participant / session / data_type
                        image_directory.mkdir(parents=True, exist_ok=True)
                        image.save(
                            image_directory / f"{participant}_{session}_{suffix}.nii.gz"
                        )
        except BaseException:
            self.cleanup()
            raise

    def __enter__(self) -> Path:
        return Path(super().__enter__())

    @staticmethod
    def _check_component(value: str, pattern: str, label: str) -> None:
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            raise ValueError(f"Invalid BIDS {label}: {value!r}")
