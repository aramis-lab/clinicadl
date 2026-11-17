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
    """Create a temporary toy :term:`BIDS` directory for examples and tests.

    The directory is created at initialization and removed on context exit or
    :meth:`cleanup`, just like :class:`tempfile.TemporaryDirectory`. Entering the
    context returns a :class:`pathlib.Path`. Setup failures also remove it.

    Parameters
    ----------
    description : dict
        JSON-serializable dataset description, written to
        ``dataset_description.json``.
    metadata : Optional[pandas.DataFrame], default=None
        Participant/session metadata, written to ``metadata.tsv``. Must contain
        ``participant_id`` and ``session_id`` columns. Without images, no
        participant/session directories are created automatically (see ``with_images``).
    with_images : Optional[dict[str, str]], default=None
        To generate zero-valued compressed NIfTI images for each participant/session couple in ``metadata``.
        If ``None``, no images will be generated.
        This argument maps data types to filename endings, e.g. ``with_images={"anat": "T1w", "pet": "trc-18FFDG_pet"}``
        will create the files ``sub-<participant_id>/ses-<session_id>/anat/sub-<participant_id>_ses-<session_id>_T1w.nii.gz`` and ``sub-<participant_id>/ses-<session_id>/pet/sub-<participant_id>_ses-<session_id>_trc-18FFDG_pet.nii.gz``.
        Data types and filename endings must follow the :bids:`BIDS syntax <common-principles.html#filenames>`.
    image_shape : tuple[int, int, int, int], default=(1, 8, 8, 8)
        When ``with_images`` is passed, image shape with channel dimension first. In particular,
        ``(1, 8, 8, 8)`` produces a scalar image with spatial shape ``(8, 8, 8)``.

    Notes
    -----
    These constant-valued images are for demonstrating APIs, not brain scans or
    training data. This helper does not validate a complete BIDS dataset.

    Examples
    ---------
    .. code-block:: python

        import pandas as pd
        import torchio as tio
        from clinicadl.io.bids import TemporaryBidsDir

        description = {
            "Name": "Toy image-loading example",
            "BIDSVersion": "1.10.0",
            "DatasetType": "raw",
        }
        metadata = pd.DataFrame(
            {
                "participant_id": ["sub-001", "sub-002"],
                "session_id": ["ses-M000", "ses-M000"],
                "age": [55.0, 62.0],
                "diagnosis": ["control", "patient"],
            }
        )
        with TemporaryBidsDir(
            description,
            metadata,
            with_images={"anat": "T1w", "pet": "trc-18FFDG_pet"},
            image_shape=(1, 4, 4, 4),
        ) as bids_directory:
            print(
                (
                    bids_directory
                    / "sub-001"
                    / "ses-M000"
                    / "anat"
                    / "sub-001_ses-M000_T1w.nii.gz"
                ).exists()
            )
            print(
                tio.ScalarImage(
                    path=bids_directory
                    / "sub-001"
                    / "ses-M000"
                    / "pet"
                    / "sub-001_ses-M000_trc-18FFDG_pet.nii.gz"
                ).shape
            )

    Outputs:

    .. code-block:: python

        True
        (1, 4, 4, 4)
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
