from __future__ import annotations

from pathlib import Path

from clinicadl.utils.dictionary.suffixes import PTH, TAR
from clinicadl.utils.dictionary.words import (
    BEST,
    CHECKPOINTS,
    FINAL,
    METRICS,
    MODEL,
    MODELS,
    VALIDATION,
)

from ....base import Directory
from ...utils import CollectionOfDirs, EpochsDir, MetricsDir, ModelDir


class TrainingModelDir(ModelDir):
    def __init__(self, path: Path):
        super().__init__(path)
        self._validation_metrics = MetricsDir(
            path=self.path / f"{VALIDATION}_{METRICS}"
        )

    @property
    def model_pt(self) -> Path:
        return (self.path / MODEL).with_suffix(PTH + TAR)

    @property
    def validation_metrics(self) -> MetricsDir:
        return self._validation_metrics


class BestModelsDir(CollectionOfDirs[TrainingModelDir, str]):
    _item_key = BEST
    _dir_type = TrainingModelDir

    def __init__(self, path: Path):
        super().__init__(path)
        self._metrics: dict[str, TrainingModelDir] = {}

    @property
    def metrics(self) -> dict[str, TrainingModelDir]:
        return self._metrics

    @property
    def metrics_list(self) -> list[str]:
        return self._items_list

    def create_metric(
        self, metric: str, overwrite: bool = False, exist_ok: bool = False
    ) -> None:
        self._create_item(metric, overwrite=overwrite, exist_ok=exist_ok)

    @classmethod
    def _items_dict_private_name(cls) -> str:
        return "_" + METRICS


class CheckpointsDir(EpochsDir[TrainingModelDir]):
    _dir_type = TrainingModelDir


class ModelsDir(Directory):
    def __init__(self, path: Path):
        super().__init__(path)
        self._best_models = BestModelsDir(path=self.path / f"{BEST}_{MODELS}")
        self._checkpoints = CheckpointsDir(path=self.path / CHECKPOINTS)
        self._final = TrainingModelDir(path=self.path / FINAL)

    @property
    def best_models(self) -> BestModelsDir:
        return self._best_models

    @property
    def checkpoints(self) -> CheckpointsDir:
        return self._checkpoints

    @property
    def final(self) -> TrainingModelDir:
        return self._final

    def get_checkpoint_dir(self, checkpoint_name: str) -> TrainingModelDir:
        """
        To get the directory of a model checkpoint from a descriptive name of this
        checkpoint.

        Parameters
        ----------
        checkpoint_name : str
            The name describing the checkpoint. Must be like:

            - ``"best-<metric>"``: to refer to the best model according to the metric ``<metric>``;
            - ``"epoch-<epoch>"``: to refer to the model at epoch ``<epoch>``;
            - ``"final"``: to refer to the model at the end of the training phase.

        Returns
        -------
        TrainingModelDir
            The :py:class:`clinicadl.io.base.Directory` associated to the checkpoint.
        """
        self.read()

        if checkpoint_name.startswith(self.best_models._item_key):
            metric = checkpoint_name.split(self.best_models._separator)[-1]
            try:
                return self.best_models.metrics[metric]
            except KeyError as e:
                raise KeyError(
                    f"No checkpoint associated to the metric '{metric}' in {str(self.best_models.path)}"
                ) from e

        elif checkpoint_name.startswith(self.checkpoints._item_key):
            epoch = checkpoint_name.split(self.checkpoints._separator)[-1]
            try:
                return self.checkpoints.epochs[int(epoch)]
            except (KeyError, ValueError) as e:
                raise KeyError(
                    f"No checkpoint associated to epoch {epoch} in {str(self.checkpoints.path)}"
                ) from e

        elif checkpoint_name == FINAL:
            return self.final
        else:
            raise ValueError(
                f"The name of the checkpoint must be like "
                f"'{self.best_models._item_key}{self.best_models._separator}...', "
                f"'{self.checkpoints._item_key}{self.checkpoints._separator}...' "
                f"or '{FINAL}'. Got: {checkpoint_name}"
            )
