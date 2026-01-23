from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.data.datasets.factory import get_dataset_from_json
from clinicadl.utils.exceptions import CannotReadJsonFieldError

if TYPE_CHECKING:
    from clinicadl.data.datasets import Dataset
    from clinicadl.data.datasets.base import BaseDataset


def get_old_dataloader(
    json_path: Path, new_dataloader: Optional[DataLoaderConfig] = None
) -> tuple[Optional[DataLoaderConfig], list[str]]:
    """
    Gets a old dataloader
    """
    try:
        DataLoaderConfig.from_json(json_path), []
    except CannotReadJsonFieldError as e:
        if new_dataloader:
            try:
                datalaoder = DataLoaderConfig.from_json(
                    json_path,
                    **{
                        arg: getattr(new_dataloader, arg) for arg in e.error.field_names
                    },
                )
                return datalaoder, e.error.field_names
            except CannotReadJsonFieldError:
                pass

    return None, []


def _get_old_dataset(
    json_path: Path, new_dataset: Optional[BaseDataset] = None
) -> tuple[Optional[Dataset], list[str]]:
    try:
        return get_dataset_from_json(json_path), []
    except CannotReadJsonFieldError as e:
        if new_dataset:
            try:
                dataset = get_dataset_from_json(
                    json_path,
                    **{
                        arg: getattr(new_dataset.config, arg)
                        for arg in e.error.field_names
                    },
                )
                return dataset, e.error.field_names
            except CannotReadJsonFieldError:
                pass
    except ValueError:
        pass

    return None, []
