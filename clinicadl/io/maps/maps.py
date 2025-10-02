from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from pandas import read_csv
from torch import load as torch_load

from clinicadl.dictionary.suffixes import JSON, LOG, TAR, TSV, TXT
from clinicadl.dictionary.utils import SEP
from clinicadl.dictionary.words import (
    ARCHITECTURE,
    ENVIRONMENT,
    METRICS,
    MODEL,
    PREDICTIONS,
    SUMMARY,
    TORCH,
    TRAINING,
)
from clinicadl.utils.json import read_json
from clinicadl.utils.typing import PathType

from .base import Directory
from .predictions import PredictionsDir
from .training import TrainingDir


class Maps(Directory):
    """
    Python equivalent of a ``MAPS`` directory produced by a :py:class:`Trainer <clinicadl.train.Trainer>`.

    A ``MAPS`` directory is the core of ``ClinicaDL``'s experiment management. It contains
    all the outputs produced by a training phase (models, metrics, logs, etc.), as well as all
    the parameter used for training. Finally, it also contains the results of inferences
    on test datasets.

    Given that a ``MAPS`` directory is quite large and can be tedious to handle, this class allows the user to
    easily access all the files stored in it.

    Here are the details on the files and directories that you can find in a ``MAPS`` directory, as well as their
    Python "equivalent" (``path`` → ``python equivalent``):

    .. dropdown:: **maps_dir**
        :icon: file-directory
        :color: muted

        General information on the ``MAPS`` directory.

        .. dropdown:: architecture.log → ``maps.architecture_log``
            :icon: file
            :color: light

            Details on the neural network architecture, provided by :py:meth:`ClinicaDLModel.write_architecture_log <clinicadl.models.ClinicaDLModel.write_architecture_log>`.

        .. dropdown:: environment.txt → ``maps.environment_txt``
            :icon: file
            :color: light

            The Python environment when the ``Maps`` was created.

        .. dropdown:: metrics.json → ``maps.metrics_json``
            :icon: file
            :color: light

            Details on all the :py:mod:`~clinicadl.metrics` mentioned in the ``.tsv`` DataFrames of the ``Maps`` directory.

        .. dropdown:: model.json → ``maps.model_json``
            :icon: file
            :color: light

            Details on the :py:meth:`~clinicadl.models.ClinicaDLModel`.

        .. dropdown:: summary.log → ``maps.summary_log``
            :icon: file
            :color: light

            Summary of the ``Maps`` directory.

        .. dropdown:: torchsummary.txt → ``maps.torchsummary_txt``
            :icon: file
            :color: light

            Summary of the neural network provided by `torchsummary <https://pypi.org/project/torch-summary/>`_ (image size after
            each layer, number of parameters for each layer, etc.).

        .. dropdown:: **training**
            :icon: file-directory
            :color: muted

            Information on the training phase: results, parameters, and data used.

            .. dropdown:: callback.json → ``maps.training.callbacks_json``
                :icon: file
                :color: light

                Details on the :py:mod:`~clinicadl.callbacks` used during training.

            .. dropdown:: optimization.json → ``maps.training.optimization_json``
                :icon: file
                :color: light

                Details on the :py:class:`optimization parameters <clinicadl.optim.OptimizationConfig>` used during
                training.

            .. dropdown:: **data**
                :icon: file-directory
                :color: muted

                Details on the data used during training.

                .. dropdown:: data.tsv → ``maps.training.data.data_tsv``
                    :icon: file
                    :color: light

                    List of all (participant, session) pairs used for training.

                .. dropdown:: **train**
                    :icon: file-directory
                    :color: muted

                    Details on the train sets.

                    .. dropdown:: dataloader.json → ``maps.training.data.train.dataloader_json``
                        :icon: file
                        :color: light

                        Details on the train :py:class:`dataloader <clinicadl.data.dataloader.DataLoaderConfig>`.

                    .. dropdown:: dataset.json → ``maps.training.data.train.dataset_json``
                        :icon: file
                        :color: light

                        Details on the train :py:class:`dataset <clinicadl.data.datasets>`.

                    .. dropdown:: **split-0**
                        :icon: file-directory
                        :color: muted

                        Details on the train set of split ``0``.

                        .. dropdown:: data.tsv → ``maps.training.data.train.splits[0].data_tsv``
                            :icon: file
                            :color: light

                            List of all (participant, session) pairs used for the train set of the split ``0``.

                .. dropdown:: **validation**
                    :icon: file-directory
                    :color: muted

                    Details on the validation sets.

                    .. dropdown:: dataset.json → ``maps.training.data.validation.dataset_json``
                        :icon: file
                        :color: light

                        Details on the validation :py:class:`dataset <clinicadl.data.datasets>`.

                    .. dropdown:: **split-0**
                        :icon: file-directory
                        :color: muted

                        Details on the validation set of split ``0``.

                        .. dropdown:: data.tsv → ``maps.training.data.validation.splits[0].data_tsv``
                            :icon: file
                            :color: light

                            List of all (participant, session) pairs used for the validation set of the split ``0``.

            .. dropdown:: **split-0**
                :icon: file-directory
                :color: muted

                Training results for split ``0``.

                .. dropdown:: computational.json → ``maps.training.splits[0].computational_json``
                    :icon: file
                    :color: light

                    Details on the :py:class:`computational parameters <clinicadl.train.ComputationalConfig>` used
                    during training of split ``0``.

                .. dropdown:: summary.log → ``maps.training.splits[0].summary_log``
                    :icon: file
                    :color: light

                    Summary of the training of split ``0``.

                .. dropdown:: **best-model-loss**
                    :icon: file-directory
                    :color: muted

                    Training results for the best model obtained with respect to the metric ``"loss"``.

                    .. dropdown:: model.pth.tar → ``maps.training.splits[0].best_models["loss"].model``
                        :icon: file
                        :color: light

                        The best model obtained with respect to the metric ``"loss"``. The content of the file is
                        defined by :py:meth:`ClinicaDLModel.save_checkpoint <clinicadl.models.ClinicaDLModel.save_checkpoint>`
                        (here only the weights of the neural networks are saved).

                    .. dropdown:: **validation_metrics**
                        :icon: file-directory
                        :color: muted

                        Validation metrics obtained by the best model obtained with respect to the metric ``"loss"``.

                        .. dropdown:: aggregated.tsv → ``maps.training.splits[0].best_models["loss"].validation_metrics.aggregated``
                            :icon: file
                            :color: light

                            Aggregated validation metrics.

                        .. dropdown:: details.tsv → ``maps.training.splits[0].best_models["loss"].validation_metrics.details``
                            :icon: file
                            :color: light

                            Validation metrics for each image.

                .. dropdown:: **checkpoints**
                    :icon: file-directory
                    :color: muted

                    Checkpoints saved by the user with :py:class:`clinicadl.callbacks.Checkpoint`.

                    .. dropdown:: **epoch-10**
                        :icon: file-directory
                        :color: muted

                        Checkpoint at epoch ``10``.

                        .. dropdown:: model.pth.tar → ``maps.training.splits[0].checkpoints.epochs[10].model``
                            :icon: file
                            :color: light

                            The model at epoch ``10``. The content of the file is
                            defined by :py:meth:`ClinicaDLModel.save_checkpoint <clinicadl.models.ClinicaDLModel.save_checkpoint>`
                            (here only the weights of the neural networks are saved).

                .. dropdown:: **logs**
                    :icon: file-directory
                    :color: muted

                    Logs saved during training.

                    .. dropdown:: training_loss.tsv → ``maps.training.splits[0].logs.training_loss``
                        :icon: file
                        :color: light

                        Training loss for each batch seen during training.

                    .. dropdown:: **tensorboard** → maps.training.splits[0].logs.tensorboard
                        :icon: file-directory
                        :color: muted

                        Where `TensorBoard <https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html>`_ files are saved.
                        To configure ``TensorBoard``, use :py:class:`clinicadl.callbacks.TensorBoard`.

                .. dropdown:: **tmp**
                    :icon: file-directory
                    :color: muted

                    Checkpoints saved at the end of each epoch, in order to resume training in case of failure.
                    Unlike ``checkpoints``, this directory will be emptied at the end of training.

                    .. dropdown:: **epoch-15**
                        :icon: file-directory
                        :color: muted

                        Checkpoint at the end of epoch ``15``.

                        .. dropdown:: model.pth.tar → ``maps.training.splits[0].tmp.epochs[15].model``
                            :icon: file
                            :color: light

                            The model at epoch ``15``. The content of the file is
                            defined by :py:meth:`ClinicaDLModel.save_checkpoint <clinicadl.models.ClinicaDLModel.save_checkpoint>`.

                        .. dropdown:: stop.json → ``maps.training.splits[0].tmp.epochs[15].stop_json``
                            :icon: file
                            :color: light

                            This file only contains a boolean stating whether the training was about to stop when the error occurred
                            (e.g. the :py:class:`early stopping <clinicadl.callbacks.EarlyStopping>` condition was met).

                        .. dropdown:: **callbacks** → maps.training.splits[0].tmp.epochs[15].callbacks
                            :icon: file-directory
                            :color: muted

                            Where :py:mod:`~clinicadl.callbacks`' checkpoints are saved.

                        .. dropdown:: **validation_metrics**
                            :icon: file-directory
                            :color: muted

                            Validation metrics obtained by the model at epoch ``15``.

                            .. dropdown:: aggregated.tsv → ``maps.training.splits[0].tmp.epochs[15].validation_metrics.aggregated``
                                :icon: file
                                :color: light

                                Aggregated validation metrics.

                            .. dropdown:: details.tsv → ``maps.training.splits[0].tmp.epochs[15].validation_metrics.details``
                                :icon: file
                                :color: light

                                Validation metrics for each image.


        .. dropdown:: **predictions**
            :icon: file-directory
            :color: muted

            Results of inference done with :py:meth:`Trainer.predict <clinicadl.train.Trainer.predict>`.

            .. dropdown:: **group-X**
                :icon: file-directory
                :color: muted

                Results of inferences on the group ``"X"``.

                .. dropdown:: data.tsv → ``maps.predictions.groups["X"].data_tsv``
                    :icon: file
                    :color: light

                    List of all (participant, session) pairs in the group ``"X"``.

                .. dropdown:: dataset.json → ``maps.predictions.groups["X"].dataset_json``
                    :icon: file
                    :color: light

                    Details on the :py:class:`dataset <clinicadl.data.datasets>` used
                    for the group ``"X"``.

                .. dropdown:: **split-0**
                    :icon: file-directory
                    :color: muted

                    Results of inference on the group ``"X"`` obtained with the models trained on split ``0`` (i.e.
                    the models in ``maps_dir/training/split-0``).

                    .. dropdown:: **best-model-loss**
                        :icon: file-directory
                        :color: muted

                        Results obtained with the best model with respect to the metric ``"loss"`` (i.e.
                        the model in ``maps_dir/training/split-0/best-model-loss``).

                        .. dropdown:: **metrics**
                            :icon: file-directory
                            :color: muted

                            Metrics on the group ``"X"``.

                            .. dropdown:: aggregated.tsv → ``maps.predictions.groups["X"].splits[0].best_models["loss"].metrics.aggregated``
                                :icon: file
                                :color: light

                                Aggregated metrics.

                            .. dropdown:: details.tsv → ``maps.predictions.groups["X"].splits[0].best_models["loss"].metrics.details``
                                :icon: file
                                :color: light

                                Metrics for each image.

    Examples
    --------

    Let's say we have a ``MAPS`` directory in ``maps_dir``:

    .. code-block:: python

        from clinicadl.io import Maps

        maps = Maps("maps_dir")
        maps.read()

    To access the checkpoint of our neural network trained on the split ``0``
    at the end of the :math:`10^{th}` epoch:

    .. code-block:: python

        >>> maps.training.splits[0].checkpoints.epochs[10].model
        PosixPath('maps_dir/training/split-0/checkpoints/epoch-10/model.pth.tar')

    To get the list of all saved checkpoints:

    .. code-block:: python

        >>> maps.training.splits[0].checkpoints.epochs_list
        [0, 5, 10]
    """

    def __init__(self, path: PathType):
        super().__init__(path)
        self._training = TrainingDir(path=self.path / TRAINING)
        self._predictions = PredictionsDir(path=self.path / PREDICTIONS)

    @property
    def training(self) -> TrainingDir:
        """Directory containing the information on the training of the model."""
        return self._training

    @property
    def predictions(self) -> PredictionsDir:
        """Directory containing the information of predictions performed with the model."""
        return self._predictions

    @property
    def architecture_log(self) -> Path:
        return (self.path / ARCHITECTURE).with_suffix(LOG)

    @property
    def model_json(self) -> Path:
        return (self.path / MODEL).with_suffix(JSON)

    @property
    def metrics_json(self) -> Path:
        return (self.path / METRICS).with_suffix(JSON)

    @property
    def environment_txt(self) -> Path:
        return (self.path / ENVIRONMENT).with_suffix(TXT)

    @property
    def summary_log(self) -> Path:
        return (self.path / SUMMARY).with_suffix(LOG)

    @property
    def torchsummary_txt(self) -> Path:
        return (self.path / (TORCH + SUMMARY)).with_suffix(TXT)

    def create(self, overwrite: bool = False, exist_ok: bool = False) -> None:
        """
        Creates the ``MAPS`` directory.

        Parameters
        ----------
        overwrite : bool, default=False
            Whether to overwrite the current directory.
        exists_ok : bool, default=False
            If the directory already exists and ``overwrite=False``, the function succeeds when ``exist_ok=True``.
        """
        super().create(overwrite=overwrite, exist_ok=exist_ok)
        self._write_environment_txt()
        self._create_summary_log()

    @staticmethod
    def read_file(path: PathType) -> Any:
        """
        To read any file inside the ``MAPS`` directory.

        Parameters
        ----------
        path : PathType
            The path in the ``MAPS`` directory.

        Returns
        -------
        Any
            The content of the file.

        Raises
        ------
        IsADirectoryError
            If the path is not a file.

        Examples
        --------

        .. code-block:: python

            from clinicadl.io import Maps

            maps = Maps("maps_dir")
            maps.read()

        .. code-block:: python

            >>> maps.read_file(maps.metrics_json)
            {
                "metrics": {
                    "mse": {
                        "name": "MSEMetric",
                        "get_not_nans": False,
                        "pred_key": "output",
                        "label_key": "label",
                        "postprocessing": [],
                        "reduction": "mean",
                    },
                }
            }
            >>> maps.read_file(maps.training.splits[0].checkpoints.epochs[0].model)
            OrderedDict([('conv0.weight',
              tensor([[[[ 7.2531e-03,  5.7384e-03,  1.4988e-02,  ..., -3.1380e-02,
                         -2.2103e-02,  5.8432e-02],
            ...
            >>> maps.read_file(maps.training.data.data_tsv)
                participant_id	session_id
            0	sub-001	ses-M000
            1	sub-002	ses-M000
            ...

        """
        path = Path(path)
        if not path.suffix:
            raise IsADirectoryError(f"{str(path)} is not a file!")

        if path.suffix == JSON:
            return read_json(path)
        elif path.suffix == LOG or path.suffix == TXT:
            with path.open("r", encoding="utf-8") as f:
                return f.read()
        elif path.suffix == TSV:
            return read_csv(path, sep=SEP)
        elif path.suffix == TAR:
            return torch_load(path)

    def _create_summary_log(self):
        """Create a summary log file."""

        summary = "==================== Summary Log ===================="
        summary += "\n\n"
        summary += (
            f"Date              : {datetime.now().strftime('%d %b %Y, %H:%M:%S')}"
        )
        summary += "\n"
        summary += f"Path              : {self.path.resolve()}"
        summary += "\n"

        with (self.summary_log).open(mode="w") as file:
            file.write(summary)

    def _write_environment_txt(self) -> None:
        """Writes the installed Python packages (via `pip freeze`) to `environment.txt`."""
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with (self.environment_txt).open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            with (self.environment_txt).open(mode="w") as file:
                file.write("pip freeze")


Maps.read.__doc__ = """
To read an existing ``MAPS`` directory.

``Maps`` will scan the entire directory to identify all files and subdirectories.
It first checks that all required elements are present, meaning the ``MAPS`` directory must be complete in order to be read.
``Maps`` also identifies "optional" files and directories, and keeps track of them.

For example, if you call ``maps.training.splits[1].summary_log`` before calling
``maps.read()``, it will raise a ``KeyError`` because ``Maps`` doesn't know that there is
a split ``1``. Once you will have called ``maps.read()``, ``Maps`` will know all the splits
present in the ``MAPS`` directory.

Raises
------
FileNotFoundError
    If a directory or a file expected in the ``MAPS`` directory is missing.
"""

Maps.remove.__doc__ = """
To delete the ``MAPS`` directory.

Parameters
----------
non_empty_ok : bool, default=False
    Whether to remove the directory even if it is non-empty.
"""
