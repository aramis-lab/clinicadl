from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from clinicadl.dictionary.suffixes import JSON, LOG, TXT
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
from clinicadl.utils.typing import PathType

from .base import Directory
from .predictions import PredictionsDir
from .training import TrainingDir


class Maps(Directory):
    """
    Entry point to access a ClinicaDL MAPS directory.

    The ``Maps`` class provides access to the structure and content of a MAPS
    directory, including training data, prediction results, model checkpoints,
    and associated metadata.

    Typically, users only need to call the ``load()`` method to inspect or reuse
    an already trained model directory.

    .. dropdown:: **maps_dir**
        :icon: file-directory
        :color: muted

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

                .. dropdown:: data.tsv → ``maps.training.data.data_tsv``
                    :icon: file
                    :color: light

                    List of all (participant, session) pairs used for training.

                .. dropdown:: **train**
                    :icon: file-directory
                    :color: muted

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

                        .. dropdown:: data.tsv → ``maps.training.data.train.splits[0].data_tsv``
                            :icon: file
                            :color: light

                            List of all (participant, session) pairs used for the train set of the split ``0``.

                .. dropdown:: **validation**
                    :icon: file-directory
                    :color: muted

                    .. dropdown:: dataset.json → ``maps.training.data.validation.dataset_json``
                        :icon: file
                        :color: light

                        Details on the validation :py:class:`dataset <clinicadl.data.datasets>`.

                    .. dropdown:: **split-0**
                        :icon: file-directory
                        :color: muted

                        .. dropdown:: data.tsv → ``maps.training.data.validation.splits[0].data_tsv``
                            :icon: file
                            :color: light

                            List of all (participant, session) pairs used for the validation set of the split ``0``.

            .. dropdown:: **split-0**
                :icon: file-directory
                :color: muted

                .. dropdown:: computational.json → ``maps.training.splits[0].computational_json``
                    :icon: file
                    :color: light

                    Details on the :py:class:`computational parameters <clinicadl.train.ComputationalConfig>` used
                    during training.

                .. dropdown:: summary.log → ``maps.training.splits[0].summary_log``
                    :icon: file
                    :color: light

                    Summary of the training on this split.

                .. dropdown:: **best-model-loss**
                    :icon: file-directory
                    :color: muted

                    .. dropdown:: model.pth.tar → ``maps.training.splits[0].best_models["loss"].model``
                        :icon: file
                        :color: light

                        The best model obtained with respect to the metric ``"loss"``. The content of the file is
                        defined by :py:meth:`ClinicadlModel.save_checkpoint <clinicadl.models.ClinicadlModel.save_checkpoint>`
                        (here only the weights of the neural networks are saved).

                    .. dropdown:: **validation_metrics**
                        :icon: file-directory
                        :color: muted

                        .. dropdown:: aggregated.tsv → ``maps.training.splits[0].best_models["loss"].validation_metrics.aggregated``
                            :icon: file
                            :color: light

                            Validation metrics obtained with the best model with respect to the metric ``"loss"``, trained on the
                            split ``0``.

                        .. dropdown:: details.tsv → ``maps.training.splits[0].best_models["loss"].validation_metrics.details``
                            :icon: file
                            :color: light

                            Validation metrics obtained for each image with the best model with respect to the metric ``"loss"``, trained on the
                            split ``0``.

                .. dropdown:: **checkpoints**
                    :icon: file-directory
                    :color: muted

                    .. dropdown:: **epoch-10**
                        :icon: file-directory
                        :color: muted

                        .. dropdown:: model.pth.tar → ``maps.training.splits[0].checkpoints.epochs[10].model``
                            :icon: file
                            :color: light

                            The model at epoch ``10``. The content of the file is
                            defined by :py:meth:`ClinicadlModel.save_checkpoint <clinicadl.models.ClinicadlModel.save_checkpoint>`
                            (here only the weights of the neural networks are saved).

                            To configure when are saved checkpoints, use :py:class:`clinicadl.callbacks.Checkpoint`.

                .. dropdown:: **logs**
                    :icon: file-directory
                    :color: muted

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



        .. dropdown:: **predictions**
            :icon: file-directory
            :color: muted


    Attributes
    ----------
    training : :py:class:`~clinicadl.io.maps.training.TrainingDir`
        Access to training-related files (data, splits, checkpoints).
    predictions : :py:class:`~clinicadl.io.maps.predictions.PredictionsDir`
        Access to prediction results for test groups.

    Examples
    --------
    .. code-block:: python

        from clinicadl.io.maps import Maps
        maps = Maps("/path/to/maps_dir")
        maps.load()  # Load existing structure

        maps.training.split_list
        >>> [0, 1, 2, 3, 4]

        first_split = maps.training.splits[0]
        print(first_split.logs.training_tsv)
        >>> /path/to/maps_dir/training/split-0/logs/training.tsv

        pred_group = maps.predictions.groups["CNvsAD"]
        metrics_file = pred_group.splits[0].best_metrics["loss"].metrics_tsv
        print(metrics_file)
        >>> /path/to/maps_dir/predictions/testCNvsAD/split-0/best-loss/metrics.tsv
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
        Creates the directory if it does not already exist.

        Parameters
        ----------
        overwrite : bool, default=False
            Whether to overwrite the current directory.
        exists_ok : bool, default=False
            If the file already exists and ``overwrite=False``, the function succeeds when ``exist_ok=True``.
        """
        super().create(overwrite=overwrite, exist_ok=exist_ok)
        self._write_environment_txt()
        self._create_summary_log()

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
        Load the MAPS directory structure from disk.

        This method reads all subfolders (training, predictions, data splits, etc.)
        and reconstructs the directory tree as Python objects.

        After loading, you can navigate through the :class:`~clinicadl.io.maps.Maps` object to access all components.

        Directory Layout
        ----------------
        Example structure after training:

        .. code-block:: text

            maps_path/
            ├── architecture.log
            ├── environment.txt
            ├── model.json
            ├── summary.log
            ├── predictions
            │   └── test<GroupName>
            │       ├── data.tsv
            │       ├── metrics.json
            │       ├── caps_dataset.json
            │       └── split-<N>
            │           ├── best-<metric>
            │           │   ├── metrics.tsv
            │           │   └── caps_output/
            │           └── computational.json
            └── training
                ├── data
                │   ├── data.tsv
                │   ├── caps_dataset.json
                │   ├── train
                │   │   └── split-<N>
                │   │       └── data.tsv
                │   └── validation
                │       └── split-<N>
                │           └── data.tsv
                ├── split-<N>
                │   ├── best-<metric>
                │   │   ├── model.pth.tar
                │   │   └── optimizer.pth.tar
                │   ├── checkpoints
                │   │   └── epoch-<K>
                │   │       ├── model.pth.tar
                │   │       └── optimizer.pth.tar
                │   ├── logs
                │   │   └── training.tsv
                │   └── tmp
                │       ├── model.pth.tar
                │       └── optimizer.pth.tar
                ├── computational.json
                ├── optimization.json
                ├── metrics.json
                └── callbacks.json

        .. note::
            - ``<N>`` refers to the split index (e.g., 0, 1, 2).
            - ``<GroupName>`` refers to the name of the test group (e.g., ``CNvsAD``).
            - ``<metric>`` refers to the metric used to select the best model (e.g., ``loss``).

        """


MAPS_DESCRIPTION = """
    .. dropdown:: **maps_dir**
        :icon: file-directory
        :color: muted

        .. dropdown:: architecture.log → ``maps.architecture_log``
            :icon: file
            :color: light

            Details on the neural network architecture, provided by :py:meth:`ClinicaDLModel.write_architecture_log <clinicadl.models.ClinicaDLModel.write_architecture_log>`.

        .. dropdown:: environment.txt → ``maps.environment_txt``
            :icon: file
            :color: light

            The Python environment when the ``Maps`` was created.

        .. dropdown:: model.json → ``maps.model_json``
            :icon: file
            :color: light

            Details on the :py:meth:`~clinicadl.models.ClinicaDLModel`.

        .. dropdown:: summary.log → ``maps.summary_log``
            :icon: file
            :color: light

            Summary of the ``Maps`` directory.
            Summary of the neural network provided by `torchsummary <https://pypi.org/project/torch-summary/>`_.

        .. dropdown:: torchsummary.txt → ``maps.torchsummary_txt``
            :icon: file
            :color: light

            Summary of the neural network provided by `torchsummary <https://pypi.org/project/torch-summary/>`_ (image size after
            each layer, number of parameters for each layer, etc.).

        .. dropdown:: **training**
            :icon: file-directory
            :color: muted

            .. dropdown:: callback.json → ``maps.model_json``
                :icon: file
                :color: light

                Details on the :py:meth:`~clinicadl.models.ClinicaDLModel`.

            .. dropdown:: model.json → ``maps.model_json``
                :icon: file
                :color: light

                Details on the :py:meth:`~clinicadl.models.ClinicaDLModel`.

            .. dropdown:: model.json → ``maps.model_json``
                :icon: file
                :color: light

                Details on the :py:meth:`~clinicadl.models.ClinicaDLModel`.

        .. dropdown:: **predictions**
            :icon: file-directory
            :color: muted
"""
