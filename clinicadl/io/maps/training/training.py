from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import JSON
from clinicadl.utils.dictionary.words import (
    DATA,
    FINAL,
    OPTIMIZATION,
)

from ...utils import mandatory
from ..utils import SplitsDir
from .data import TrainingDataDir
from .splits import TrainingSplitDir
from .splits.models import BestModelsDir, CheckpointsDir, TrainingModelDir

SEPARATOR = "_"


class TrainingDir(SplitsDir[TrainingSplitDir]):
    _dir_type = TrainingSplitDir

    def __init__(self, path: Path):
        super().__init__(path)
        self._data = TrainingDataDir(path=self.path / DATA)

    @property
    def data(self) -> TrainingDataDir:
        return self._data

    @property
    @mandatory
    def optimization_json(self) -> Path:
        return (self.path / OPTIMIZATION).with_suffix(JSON)

    def create_split(
        self, split_idx: int, overwrite: bool = False, exist_ok: bool = False
    ) -> None:
        super().create_split(split_idx, overwrite=overwrite, exist_ok=exist_ok)
        self.data.train.create_split(split_idx, overwrite=overwrite, exist_ok=exist_ok)
        self.data.validation.create_split(
            split_idx, overwrite=overwrite, exist_ok=exist_ok
        )

    def delete_split(self, split_idx: int) -> None:
        split_exists = False
        for dir in [super(), self.data.train, self.data.validation]:
            try:
                dir.delete_split(split_idx)
            except FileNotFoundError:
                continue
            else:
                split_exists = True
        if not split_exists:
            raise FileNotFoundError(
                f"No mention of split {split_idx} found in {self.path}"
            )

    def read(self) -> None:
        super().read()

        for split in self.splits_list:
            if split not in self._data._train.splits_list:
                raise FileNotFoundError(
                    f"split-{split} not found in the training data ({str(self._data._train.path)})"
                )
            if split not in self._data._validation.splits_list:
                raise FileNotFoundError(
                    f"split-{split} not found in the validation data ({str(self._data._validation.path)})"
                )

    def get_checkpoint_dir(self, checkpoint_name: str) -> TrainingModelDir:
        """
        To get the directory of a model checkpoint from a descriptive name of this
        checkpoint.

        Parameters
        ----------
        checkpoint_name : str
            The name describing the checkpoint. Must be like:

            - ``"split-<split>_best-<metric>"``: to refer to the best model trained on split ``<split>`` according to the metric ``<metric>``;
            - ``"split-<split>_epoch-<epoch>"``: to refer to the model trained on split ``<split>`` at epoch ``<epoch>``;
            - ``"split-<split>_final"``: to refer to the model at the end of the training on split ``<split>``.

        Returns
        -------
        TrainingModelDir
            The :py:class:`clinicadl.io.base.Directory` associated to the checkpoint.
        """
        split, name = self.read_checkpoint_name(checkpoint_name)

        return self.splits[split].models.get_checkpoint_dir(name)

    def read_checkpoint_name(self, checkpoint_name: str) -> tuple[int, str]:
        """
        To read the name of a model checkpoint.

        Parameters
        ----------
        checkpoint_name : str
            The name describing the checkpoint. Must be like:

            - ``"split-<split>_best-<metric>"``: to refer to the best model trained on split ``<split>`` according to the metric ``<metric>``;
            - ``"split-<split>_epoch-<epoch>"``: to refer to the model trained on split ``<split>`` at epoch ``<epoch>``;
            - ``"split-<split>_final"``: to refer to the model at the end of the training on split ``<split>``.

        Returns
        -------
        tuple[int, str]
            The split associated the this checkpoint, and its name inside the split.
        """
        if checkpoint_name.startswith(self._item_key):
            try:
                split, name = checkpoint_name.split(SEPARATOR)
            except ValueError:
                pass
            else:
                split_idx = split.split(self._separator)[-1]
                try:
                    split_dir = self.splits[int(split_idx)]
                except (ValueError, KeyError) as e:
                    raise KeyError(
                        f"No checkpoint associated to split {split_idx} in {str(self.path)}"
                    ) from e
                try:
                    split_dir.models.get_checkpoint_dir(name)
                except ValueError:
                    pass
                else:
                    return int(split_idx), name

        raise ValueError(
            "The name of the checkpoint must be like "
            f"'{self._item_key}{self._separator}...{SEPARATOR}{BestModelsDir._item_key}{BestModelsDir._separator}...', "
            f"'{self._item_key}{self._separator}...{SEPARATOR}{CheckpointsDir._item_key}{CheckpointsDir._separator}...' "
            f"or '{self._item_key}{self._separator}...{SEPARATOR}{FINAL}'. Got: {checkpoint_name}"
        )
