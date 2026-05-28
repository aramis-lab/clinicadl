from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from torch import load as torch_load
from torch import save as torch_save

from clinicadl.utils.dictionary.suffixes import JSON, LOG, PT, TSV, TXT
from clinicadl.utils.dictionary.utils import SEP
from clinicadl.utils.dictionary.words import (
    ARCHITECTURE,
    CALLBACKS,
    ENVIRONMENT,
    EXEC,
    METRICS,
    MODEL,
    NN,
    PARTICIPANT_ID,
    PREDICTION,
    SESSION_ID,
    SUMMARY,
    TEST,
    TRAINING,
)
from clinicadl.utils.json import read_json, write_json
from clinicadl.utils.typing import PathType

from ..base import Directory
from ..utils import mandatory
from .exec import ExecDir
from .inference import PredictionDir, TestDir
from .summary import MapsSummary
from .training import TrainingDir


class Maps(Directory):
    """
    Python equivalent of a ``MAPS`` directory produced by a :py:class:`Trainer <clinicadl.train.Trainer>`.

    A ``MAPS`` directory is the core of ``ClinicaDL``'s experiment management. It contains
    all the outputs produced by a training phase (models, metrics, logs, etc.), as well as all
    the hyperparameters used for training. Finally, it also contains the results of test and inference
    on new datasets.

    Given that a ``MAPS`` directory is quite large and can be tedious to handle, this class allows the user to
    easily access all the files stored in it.

    Here are the details of the files and subdirectories that you can find in a ``MAPS`` directory, as well as their
    Python "equivalent" (``path`` → ``python equivalent``):

    .. dropdown:: **maps_dir**
        :icon: file-directory
        :color: muted

        General information on the ``MAPS`` directory.

        .. dropdown:: architecture.log → ``maps.architecture_log``
            :icon: file
            :color: light

            Details on the neural network architecture.

        .. dropdown:: environment.txt → ``maps.environment_txt``
            :icon: file
            :color: light

            The Python environment when the ``Maps`` was created.

        .. dropdown:: callbacks.json → ``maps.callbacks_json``
            :icon: file
            :color: light

            Details on the :py:mod:`~clinicadl.callbacks` used.

        .. dropdown:: metrics.json → ``maps.metrics_json``
            :icon: file
            :color: light

            Details on all the :py:mod:`~clinicadl.metrics` computed.

        .. dropdown:: model.json → ``maps.model_json``
            :icon: file
            :color: light

            Details on the :py:meth:`~clinicadl.models.Model`.

        .. dropdown:: nn_summary.txt → ``maps.nn_summary_txt``
            :icon: file
            :color: light

            Summary of the neural network provided by :py:meth:`Model.get_summary <clinicadl.models.Model.get_summary>`.

        .. dropdown:: summary.log → ``maps.summary_log``
            :icon: file
            :color: light

            Summary of the ``Maps`` directory.

        .. dropdown:: **training**
            :icon: file-directory
            :color: muted

            Here are stored the outputs and the information related to :py:meth:`Trainer.train <clinicadl.train.Trainer.train>` and
            :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`.

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

                    .. dropdown:: **split-0**
                        :icon: file-directory
                        :color: muted

                        Details on the train set of split ``0``.

                        .. dropdown:: data.tsv → ``maps.training.data.train.splits[0].data_tsv``
                            :icon: file
                            :color: light

                            List of all (participant, session) pairs used for the train set of the split ``0``.

                        .. dropdown:: dataloader.json → ``maps.training.data.train.splits[0].dataloader_json``
                            :icon: file
                            :color: light

                            Details on the train :py:class:`dataloader <clinicadl.data.dataloader.DataLoader>`
                            for split ``0``.

                        .. dropdown:: dataset.json → ``maps.training.data.train.splits[0].dataset_json``
                            :icon: file
                            :color: light

                            Details on the train :py:class:`dataset <clinicadl.data.datasets>` for split ``0``.

                .. dropdown:: **validation**
                    :icon: file-directory
                    :color: muted

                    Details on the validation sets.

                    .. dropdown:: **split-0**
                        :icon: file-directory
                        :color: muted

                        Details on the validation set of split ``0``.

                        .. dropdown:: data.tsv → ``maps.training.data.validation.splits[0].data_tsv``
                            :icon: file
                            :color: light

                            List of all (participant, session) pairs used for the validation set of the split ``0``.

                        .. dropdown:: dataloader.json → ``maps.training.data.train.splits[0].dataloader_json``
                            :icon: file
                            :color: light

                            Details on the validation :py:class:`dataloader <clinicadl.data.dataloader.DataLoader>`
                            for split ``0``.

                        .. dropdown:: dataset.json → ``maps.training.data.validation.splits[0].dataset_json``
                            :icon: file
                            :color: light

                            Details on the validation :py:class:`dataset <clinicadl.data.datasets>` for split ``0``.

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

                    Summary of the training of split ``0``, along with computational metrics provided by
                    :py:mod:`~clinicadl.callbacks.MonitorCallback`.

                .. dropdown:: warning.log → ``maps.training.splits[0].warning_log``
                    :icon: file
                    :color: light

                    Potential warning logs saved by :py:class:`~clinicadl.callbacks.LoggerCallback` (if ``save_logs=True``) that
                    have been raised when training.

                .. dropdown:: **logs**
                    :icon: file-directory
                    :color: muted

                    Logs saved during training.

                    .. dropdown:: training_loss.tsv → ``maps.training.splits[0].logs.training_loss_tsv``
                        :icon: file
                        :color: light

                        Training loss for each batch seen during training.

                    .. dropdown:: computational.tsv → ``maps.training.splits[0].logs.computational_tsv``
                        :icon: file
                        :color: light

                        Computational metrics saved by :py:mod:`~clinicadl.callbacks.MonitorCallback`.

                    .. dropdown:: **learning_rates** → maps.training.splits[0].logs.learning_rates
                        :icon: file-directory
                        :color: muted

                        If one or several :py:mod:`~clinicadl.callbacks.LRSchedulerCallback` are activated,
                        this is where the history of the learning rates will be saved.

                .. dropdown:: **models**
                    :icon: file-directory
                    :color: muted

                    All the models saved during training.

                    .. dropdown:: **best_models**
                        :icon: file-directory
                        :color: muted

                        Best models obtained with respect to the metrics monitored in :py:class:`~clinicadl.callbacks.ModelCheckpointCallback`.

                        .. dropdown:: **best-mse**
                            :icon: file-directory
                            :color: muted

                            Results for the best model obtained with respect to the metric ``"mse"``.

                            .. dropdown:: warning.log → ``maps.training.splits[0].models.best_models.metrics["mse"].warning_log``
                                :icon: file
                                :color: light

                                Potential warning logs saved by :py:class:`~clinicadl.callbacks.LoggerCallback` (if ``save_logs=True``) that
                                have been raised when evaluating this model.

                            .. dropdown:: model.pt → ``maps.training.splits[0].models.best_models.metrics["mse"].model_pt``
                                :icon: file
                                :color: light

                                The weights of the model.

                            .. dropdown:: **validation_metrics**
                                :icon: file-directory
                                :color: muted

                                Validation metrics obtained by the best model obtained with respect to the metric ``"mse"``.

                                .. dropdown:: aggregated.tsv → ``maps.training.splits[0].models.best_models.metrics["mse"].validation_metrics.aggregated_tsv``
                                    :icon: file
                                    :color: light

                                    Aggregated validation metrics.

                                .. dropdown:: details.tsv → ``maps.training.splits[0].models.best_models.metrics["mse"].validation_metrics.details_tsv``
                                    :icon: file
                                    :color: light

                                    Validation metrics for each image.

                    .. dropdown:: **checkpoints**
                        :icon: file-directory
                        :color: muted

                        The model at the epochs defined in :py:class:`~clinicadl.callbacks.ModelCheckpointCallback`.

                        .. dropdown:: **epoch-10**
                            :icon: file-directory
                            :color: muted

                            Results for the model at epoch ``10``.

                            .. dropdown:: warning.log → ``maps.training.splits[0].models.checkpoints.epochs[10].warning_log``
                                :icon: file
                                :color: light

                                Potential warning logs saved by :py:class:`~clinicadl.callbacks.LoggerCallback` (if ``save_logs=True``) that
                                have been raised when evaluating this model.

                            .. dropdown:: model.pt → ``maps.training.splits[0].models.checkpoints.epochs[10].model_pt``
                                :icon: file
                                :color: light

                                The weights of the model.

                            .. dropdown:: **validation_metrics**
                                :icon: file-directory
                                :color: muted

                                Validation metrics obtained by the model at epoch ``10``.

                                .. dropdown:: aggregated.tsv → ``maps.training.splits[0].models.checkpoints.epochs[10].validation_metrics.aggregated_tsv``
                                    :icon: file
                                    :color: light

                                    Aggregated validation metrics.

                                .. dropdown:: details.tsv → ``maps.training.splits[0].models.checkpoints.epochs[10].validation_metrics.details_tsv``
                                    :icon: file
                                    :color: light

                                    Validation metrics for each image.

                    .. dropdown:: **final**
                        :icon: file-directory
                        :color: muted

                        The model at the end of training, saved unless ``save_last=False`` in :py:class:`~clinicadl.callbacks.ModelCheckpointCallback`.

                        .. dropdown:: warning.log → ``maps.training.splits[0].models.final.warning_log``
                            :icon: file
                            :color: light

                            Potential warning logs saved by :py:class:`~clinicadl.callbacks.LoggerCallback` (if ``save_logs=True``) that
                            have been raised when evaluating this model.

                        .. dropdown:: model.pt → ``maps.training.splits[0].models.final.model_pt``
                            :icon: file
                            :color: light

                            The weights of the model.

                        .. dropdown:: **validation_metrics**
                            :icon: file-directory
                            :color: muted

                            Validation metrics obtained by the final model.

                            .. dropdown:: aggregated.tsv → ``maps.training.splits[0].models.final.validation_metrics.aggregated_tsv``
                                :icon: file
                                :color: light

                                Aggregated validation metrics.

                            .. dropdown:: details.tsv → ``maps.training.splits[0].models.final.validation_metrics.details_tsv``
                                :icon: file
                                :color: light

                                Validation metrics for each image.

                .. dropdown:: **tmp**
                    :icon: file-directory
                    :color: muted

                    Checkpoints saved by :py:class:`~clinicadl.callbacks.TrainingCheckpointCallback` in order to resume training in case of failure.
                    **This directory will be emptied at the end of training.**

                    .. dropdown:: **epoch-15**
                        :icon: file-directory
                        :color: muted

                        Checkpoint at the end of epoch ``15``.

                        .. dropdown:: model.pt → ``maps.training.splits[0].tmp.epochs[15].model_pt``
                            :icon: file
                            :color: light

                            The weights of the model.

                        .. dropdown:: optimizer.pt → ``maps.training.splits[0].tmp.epochs[15].optimizer_pt``
                            :icon: file
                            :color: light

                            The state of the optimizer(s).

                        .. dropdown:: scaler.pt → ``maps.training.splits[0].tmp.epochs[15].scaler_pt``
                            :icon: file
                            :color: light

                            The state of the `Gradient Scaler <https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling>`_.

                        .. dropdown:: state.json → ``maps.training.splits[0].tmp.epochs[15].state_json``
                            :icon: file
                            :color: light

                            The :py:class:`state of the Trainer <clinicadl.train.TrainerState>`.

                        .. dropdown:: **callbacks** → maps.training.splits[0].tmp.epochs[15].callbacks
                            :icon: file-directory
                            :color: muted

                            Where :py:mod:`~clinicadl.callbacks`' checkpoints are saved.

                        .. dropdown:: **validation_metrics**
                            :icon: file-directory
                            :color: muted

                            Validation metrics at epoch ``15``.

                            .. dropdown:: aggregated.tsv → ``maps.training.splits[0].tmp.epochs[15].validation_metrics.aggregated_tsv``
                                :icon: file
                                :color: light

                                Aggregated validation metrics.

                            .. dropdown:: details.tsv → ``maps.training.splits[0].tmp.epochs[15].validation_metrics.details_tsv``
                                :icon: file
                                :color: light

                                Validation metrics for each image.

                .. dropdown:: **validation_metrics**
                    :icon: file-directory
                    :color: muted

                    Validation metrics at every validation step.

                    .. dropdown:: aggregated.tsv → ``maps.training.splits[0].validation_metrics.aggregated_tsv``
                        :icon: file
                        :color: light

                        Aggregated validation metrics.

                    .. dropdown:: details.tsv → ``maps.training.splits[0].validation_metrics.details_tsv``
                        :icon: file
                        :color: light

                        Validation metrics for each image.

        .. dropdown:: **test**
            :icon: file-directory
            :color: muted

            Results of :py:meth:`Trainer.test <clinicadl.train.Trainer.test>`.

            .. dropdown:: **group-X**
                :icon: file-directory
                :color: muted

                Results of test on the group ``"X"``.

                .. dropdown:: data.tsv → ``maps.test.groups["X"].data_tsv``
                    :icon: file
                    :color: light

                    List of all (participant, session) pairs in the group ``"X"``.

                .. dropdown:: dataloader.json → ``maps.test.groups["X"].dataloader_json``
                    :icon: file
                    :color: light

                    Details on the :py:class:`dataloader <clinicadl.data.dataloader.DataLoader>` used
                    for the group ``"X"``.

                .. dropdown:: dataset.json → ``maps.test.groups["X"].dataset_json``
                    :icon: file
                    :color: light

                    Details on the :py:class:`dataset <clinicadl.data.datasets>` used
                    for the group ``"X"``.

                .. dropdown:: **results**
                    :icon: file-directory
                    :color: muted

                    .. dropdown:: **split-0**
                        :icon: file-directory
                        :color: muted

                        Results of test on the group ``"X"`` obtained with models trained on split ``0``
                        (i.e. the models in ``maps_dir/training/split-0/models``).

                        .. dropdown:: **epoch-10**
                            :icon: file-directory
                            :color: muted

                            Results obtained with the model at epoch ``10``, trained on split ``0``
                            (i.e. the model in ``maps_dir/training/split-0/models/checkpoints/epoch-10``).

                            .. dropdown:: warning.log → ``maps.test.groups["X"].results.splits[0].models["epoch-10"].warning_log``
                                :icon: file
                                :color: light

                                Potential warning logs saved by :py:class:`~clinicadl.callbacks.LoggerCallback` (if ``save_logs=True``) that
                                have been raised when testing with this model.

                            .. dropdown:: computational.json → ``maps.test.groups["X"].results.splits[0].models["epoch-10"].computational_json``
                                :icon: file
                                :color: light

                                Details on the :py:class:`computational parameters <clinicadl.train.ComputationalConfig>` used
                                when testing.

                            .. dropdown:: **metrics**
                                :icon: file-directory
                                :color: muted

                                Metrics on the group ``"X"``.

                                .. dropdown:: aggregated.tsv → ``maps.test.groups["X"].results.splits[0].models["epoch-10"].metrics.aggregated_tsv``
                                    :icon: file
                                    :color: light

                                    Aggregated metrics.

                                .. dropdown:: details.tsv → ``maps.test.groups["X"].results.splits[0].models["epoch-10"].metrics.details_tsv``
                                    :icon: file
                                    :color: light

                                    Metrics for each image.


                        .. dropdown:: **final**
                            :icon: file-directory
                            :color: muted

                            Results obtained with the final model trained on the split ``0``
                            (i.e. the model in ``maps_dir/training/split-0/models/final``).

                            .. dropdown:: warning.log → ``maps.test.groups["X"].results.splits[0].models["final"].warning_log``
                                :icon: file
                                :color: light

                                Potential warning logs saved by :py:class:`~clinicadl.callbacks.LoggerCallback` (if ``save_logs=True``) that
                                have been raised when testing with this model.

                            .. dropdown:: computational.json → ``maps.test.groups["X"].results.splits[0].models["final"].computational_json``
                                :icon: file
                                :color: light

                                Details on the :py:class:`computational parameters <clinicadl.train.ComputationalConfig>` used
                                when testing.

                            .. dropdown:: **metrics**
                                :icon: file-directory
                                :color: muted

                                Metrics on the group ``"X"``.

                                .. dropdown:: aggregated.tsv → ``maps.test.groups["X"].results.splits[0].models["final"].metrics.aggregated_tsv``
                                    :icon: file
                                    :color: light

                                    Aggregated metrics.

                                .. dropdown:: details.tsv → ``maps.test.groups["X"].results.splits[0].models["final"].metrics.details_tsv``
                                    :icon: file
                                    :color: light

                                    Metrics for each image.

                        .. dropdown:: **best-mse**
                            :icon: file-directory
                            :color: muted

                            Results obtained with the best model with respect to the metric ``"mse"``, trained on the split ``0``
                            (i.e. the model in ``maps_dir/training/split-0/models/best_models/best-mse``).

                            .. dropdown:: warning.log → ``maps.test.groups["X"].results.splits[0].models["best-mse"].warning_log``
                                :icon: file
                                :color: light

                                Potential warning logs saved by :py:class:`~clinicadl.callbacks.LoggerCallback` (if ``save_logs=True``) that
                                have been raised when testing with this model.

                            .. dropdown:: computational.json → ``maps.test.groups["X"].results.splits[0].models["best-mse"].computational_json``
                                :icon: file
                                :color: light

                                Details on the :py:class:`computational parameters <clinicadl.train.ComputationalConfig>` used
                                when testing.

                            .. dropdown:: **metrics**
                                :icon: file-directory
                                :color: muted

                                Metrics on the group ``"X"``.

                                .. dropdown:: aggregated.tsv → ``maps.test.groups["X"].results.splits[0].models["best-mse"].metrics.aggregated_tsv``
                                    :icon: file
                                    :color: light

                                    Aggregated metrics.

                                .. dropdown:: details.tsv → ``maps.test.groups["X"].results.splits[0].models["best-mse"].metrics.details_tsv``
                                    :icon: file
                                    :color: light

                                    Metrics for each image.

        .. dropdown:: **exec**
            :icon: file-directory
            :color: muted

            Logging messages saved by :py:class:`~clinicadl.callbacks.LoggerCallback` (if ``save_logs=True``).

            .. dropdown:: **run-train_2025_12_31_23_59_59**
                :icon: file-directory
                :color: muted

                Logging messages saved when :py:meth:`Trainer.train <clinicadl.train.Trainer.train>` was run
                (the end of the filename is the date).

                .. dropdown:: debug.log → ``maps.exec.runs["train_2025_12_31_23_59_59"].debug_log``
                    :icon: file
                    :color: light

                    Debug logs (if ``debug=True`` in :py:class:`~clinicadl.callbacks.LoggerCallback`).

                .. dropdown:: error.log → ``maps.exec.runs["train_2025_12_31_23_59_59"].error_log``
                    :icon: file
                    :color: light

                    Error logs.

                .. dropdown:: info.log → ``maps.exec.runs["train_2025_12_31_23_59_59"].info_log``
                    :icon: file
                    :color: light

                    Info and warning logs.

    Examples
    --------

    Let's say we have a ``MAPS`` directory in ``maps_dir``:

    .. code-block:: python

        from clinicadl.io.maps import Maps

        maps = Maps("maps_dir")
        maps.read()

    To access the checkpoint of our neural network trained on the split ``0``
    at the end of the :math:`10^{th}` epoch:

    .. code-block:: python

        >>> maps.training.splits[0].models.checkpoints.epochs[10].model_pt
        PosixPath('maps_dir/training/split-0/models/checkpoints/epoch-10/model.pt')

    To get the list of all saved checkpoints:

    .. code-block:: python

        >>> maps.training.splits[0].models.checkpoints.epochs_list
        [0, 5, 10]
    """

    def __init__(self, path: PathType):
        super().__init__(path)
        self._training = TrainingDir(path=self.path / TRAINING)
        self._test = TestDir(path=self.path / TEST)
        self._prediction = PredictionDir(path=self.path / PREDICTION)
        self._exec = ExecDir(path=self.path / EXEC)

    @property
    @mandatory
    def training(self) -> TrainingDir:
        return self._training

    @property
    def test(self) -> TestDir:
        return self._test

    @property
    def prediction(self) -> PredictionDir:
        return self._prediction

    @property
    def exec(self) -> ExecDir:
        return self._exec

    @property
    def architecture_log(self) -> Path:
        return (self.path / ARCHITECTURE).with_suffix(LOG)

    @property
    @mandatory
    def model_json(self) -> Path:
        return (self.path / MODEL).with_suffix(JSON)

    @property
    @mandatory
    def metrics_json(self) -> Path:
        return (self.path / METRICS).with_suffix(JSON)

    @property
    @mandatory
    def environment_txt(self) -> Path:
        return (self.path / ENVIRONMENT).with_suffix(TXT)

    @property
    def summary_log(self) -> Path:
        return (self.path / SUMMARY).with_suffix(LOG)

    @property
    def nn_summary_txt(self) -> Path:
        return (self.path / f"{NN}_{SUMMARY}").with_suffix(TXT)

    @property
    @mandatory
    def callbacks_json(self) -> Path:
        return (self.path / CALLBACKS).with_suffix(JSON)

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
    def open_file(path: PathType) -> Any:
        """
        To read and load any file inside the ``MAPS`` directory.

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
        FileNotFoundError
            If the path does not match any existing file.

        Examples
        --------

        .. code-block:: python

            from clinicadl.io.maps import Maps

            maps = Maps("maps_dir")
            maps.read()

        .. code-block:: python

            >>> maps.open_file(maps.metrics_json)
            {
                "metrics": {
                    "mse": {
                        "name_": "MSEMetric",
                        "get_not_nans": False,
                        "pred_key": "output",
                        "label_key": "label",
                        "postprocessing": [],
                        "reduction": "mean",
                    },
                }
            }
            >>> maps.open_file(maps.training.splits[0].models.checkpoints.epochs[1].model_pt)
            OrderedDict([('conv0.weight',
              tensor([[[[ 7.2531e-03,  5.7384e-03,  1.4988e-02,  ..., -3.1380e-02,
                         -2.2103e-02,  5.8432e-02],
            ...
            >>> maps.open_file(maps.training.data.data_tsv)
                participant_id	session_id
            0	sub-001	ses-M000
            1	sub-002	ses-M000
            ...

        See Also
        --------
        :py:meth:`save_file`

        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{str(path)} is not a file!")

        if path.suffix == JSON:
            return read_json(path)
        elif path.suffix == TSV:
            try:
                return pd.read_csv(path, sep=SEP)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
        elif path.suffix == PT:
            return torch_load(path, weights_only=True)
        else:
            with path.open("r", encoding="utf-8") as f:
                return f.read()

    @staticmethod
    def save_file(obj: Any, path: PathType, overwrite: bool = False) -> None:
        """
        To save an object in any file of the ``MAPS`` directory.

        Parameters
        ----------
        obj : Any
            The object to save.
        path : PathType
            The path in the ``MAPS`` directory.
        overwrite : bool, default=False
            To overwrite the file if it exists.

        Raises
        ------
        IsADirectoryError
            If the path is not a file path.
        FileExistsError
            If the file exists and ``overwrite=False``.
        ValueError
            If the file type is not supported. Supported file types in a ``MAPS``
            directory are ``.json``, ``.txt``, ``.log``, ``.tsv``, and ``.pt``.

        Examples
        --------

        .. code-block:: python

            from clinicadl.io.maps import Maps

            maps = Maps("maps_dir")
            maps.read()

        .. code-block:: python

            >>> metrics = {
                    "metrics": {
                        "mse": {
                            "name_": "MSEMetric",
                            "get_not_nans": False,
                            "pred_key": "output",
                            "label_key": "label",
                            "postprocessing": [],
                            "reduction": "mean",
                        },
                    }
                }
            >>> maps.save_file(metrics, maps.metrics_json)

        See Also
        --------
        :py:meth:`open_file`

        """
        path = Path(path)
        if not path.suffix:
            raise IsADirectoryError(f"{str(path)} is not a valid file name!")
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{str(path)} exists! To overwrite it, pass overwrite=True."
            )

        if path.suffix == JSON:
            write_json(path, obj, overwrite=True)
        elif path.suffix == LOG or path.suffix == TXT:
            with path.open("w", encoding="utf-8") as f:
                f.write(str(obj))
        elif path.suffix == TSV:
            assert isinstance(obj, (pd.DataFrame, pd.Series))
            obj.to_csv(path, sep=SEP, index=False)
        elif path.suffix == PT:
            torch_save(obj, path)
        else:
            raise ValueError(
                f"'{path.suffix}' files are not supported. The supported files in a MAPS directory are {[JSON, LOG, TXT, TSV, PT]}"
            )

    def delete_split(self, split_idx: int) -> None:
        """
        To properly remove a split from the ``MAPS`` directory.

        This function will delete the directory ``<maps>/training/split-<split_idx>``, will update
        ``<maps>/training/data``, and will delete all the results associated to models trained on
        this split in ``<maps>/test``.

        Parameters
        ----------
        split_idx : int
            The id of the split to delete.
        """
        self.read()
        if split_idx not in self.training.splits_list:
            raise ValueError(f"Split {split_idx} does not exist in your MAPS directory")

        for group in self.prediction.groups_list:
            if split_idx in self.prediction.groups[group].results.splits_list:
                self.prediction.groups[group].results.splits[split_idx].remove(
                    non_empty_ok=True
                )

        for group in self.test.groups_list:
            if split_idx in self.test.groups[group].results.splits_list:
                self.test.groups[group].results.splits[split_idx].remove(
                    non_empty_ok=True
                )

        self.training.data.train.splits[split_idx].remove(non_empty_ok=True)
        self.training.data.validation.splits[split_idx].remove(non_empty_ok=True)

        self.training.splits[split_idx].remove(non_empty_ok=True)

        self._update_training_data()

    def _update_training_data(self):
        """
        To update the DataFrame listing the training and validation (participant, session) pairs.
        """
        self.training.read()
        dfs = []
        for split_idx in self.training.splits_list:
            dfs.append(
                self.open_file(self.training.data.train.splits[split_idx].data_tsv)
            )
            dfs.append(
                self.open_file(self.training.data.validation.splits[split_idx].data_tsv)
            )

        df = pd.concat(dfs).drop_duplicates().sort_values([PARTICIPANT_ID, SESSION_ID])

        self.save_file(df, self.training.data.data_tsv, overwrite=True)

    def _create_summary_log(self):
        """Create a summary log file."""
        MapsSummary(self.summary_log).create()

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
It first checks that all required elements are present, meaning that the ``MAPS`` directory must be complete in order to be read.
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
