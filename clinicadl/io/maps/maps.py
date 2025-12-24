from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from torch import load as torch_load
from torch import save as torch_save

from clinicadl.utils.dictionary.suffixes import JSON, LOG, PTH, TAR, TSV, TXT
from clinicadl.utils.dictionary.utils import SEP
from clinicadl.utils.dictionary.words import (
    ARCHITECTURE,
    ENVIRONMENT,
    METRICS,
    MODEL,
    NN,
    PREDICTION,
    SUMMARY,
    TEST,
    TRAINING,
)
from clinicadl.utils.json import read_json, write_json
from clinicadl.utils.typing import PathType

from ..base import Directory
from .inference import PredictionDir, TestDir
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

            Details on the neural network architecture, provided by :py:meth:`Model.get_architecture <clinicadl.models.Model.get_architecture>`.

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

            Details on the :py:meth:`~clinicadl.models.Model`.

        .. dropdown:: summary.log → ``maps.summary_log``
            :icon: file
            :color: light

            Summary of the ``Maps`` directory.

        .. dropdown:: nn_summary.txt → ``maps.nn_summary_txt``
            :icon: file
            :color: light

            Summary of the neural network provided by :py:meth:`Model.get_summary <clinicadl.models.Model.get_summary>`.

        .. dropdown:: **training**
            :icon: file-directory
            :color: muted

            Here are stored the outputs and the information related to :py:meth:`Trainer.train <clinicadl.train.Trainer.train>` and
            :py:meth:`Trainer.validate <clinicadl.train.Trainer.validate>`.

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

                .. dropdown:: **models**
                    :icon: file-directory
                    :color: muted

                    All the models saved during training.

                    .. dropdown:: **best_models**
                        :icon: file-directory
                        :color: muted

                        Best models obtained with respect to the metrics monitored in :py:class:`~clinicadl.callbacks.Checkpoint`.

                        .. dropdown:: **best-mse**
                            :icon: file-directory
                            :color: muted

                            Results for the best model obtained with respect to the metric ``"mse"``.

                            .. dropdown:: model.pth.tar → ``maps.training.splits[0].models.best_models.metrics["mse"].model``
                                :icon: file
                                :color: light

                                The weights of the model.

                            .. dropdown:: **validation_metrics**
                                :icon: file-directory
                                :color: muted

                                Validation metrics obtained by the best model obtained with respect to the metric ``"mse"``.

                                .. dropdown:: aggregated.tsv → ``maps.training.splits[0].models.best_models.metrics["mse"].validation_metrics.aggregated``
                                    :icon: file
                                    :color: light

                                    Aggregated validation metrics.

                                .. dropdown:: details.tsv → ``maps.training.splits[0].models.best_models.metrics["mse"].validation_metrics.details``
                                    :icon: file
                                    :color: light

                                    Validation metrics for each image.

                    .. dropdown:: **checkpoints**
                        :icon: file-directory
                        :color: muted

                        The model at the epochs defined in :py:class:`~clinicadl.callbacks.Checkpoint`.

                        .. dropdown:: **epoch-10**
                            :icon: file-directory
                            :color: muted

                            Results for the model at epoch ``10``.

                            .. dropdown:: model.pth.tar → ``maps.training.splits[0].models.checkpoints.epochs[10].model``
                                :icon: file
                                :color: light

                                The weights of the model.

                            .. dropdown:: **validation_metrics**
                                :icon: file-directory
                                :color: muted

                                Validation metrics obtained by the model at epoch ``10``.

                                .. dropdown:: aggregated.tsv → ``maps.training.splits[0].models.checkpoints.epochs[10].validation_metrics.aggregated``
                                    :icon: file
                                    :color: light

                                    Aggregated validation metrics.

                                .. dropdown:: details.tsv → ``maps.training.splits[0].models.checkpoints.epochs[10].validation_metrics.details``
                                    :icon: file
                                    :color: light

                                    Validation metrics for each image.

                    .. dropdown:: **final**
                        :icon: file-directory
                        :color: muted

                        The model at the end of training.

                        .. dropdown:: model.pth.tar → ``maps.training.splits[0].models.final.model``
                            :icon: file
                            :color: light

                            The weights of the model.

                        .. dropdown:: **validation_metrics**
                            :icon: file-directory
                            :color: muted

                            Validation metrics obtained by the final model.

                            .. dropdown:: aggregated.tsv → ``maps.training.splits[0].models.final.validation_metrics.aggregated``
                                :icon: file
                                :color: light

                                Aggregated validation metrics.

                            .. dropdown:: details.tsv → ``maps.training.splits[0].models.final.validation_metrics.details``
                                :icon: file
                                :color: light

                                Validation metrics for each image.

                .. dropdown:: **tmp**
                    :icon: file-directory
                    :color: muted

                    Checkpoints saved at the end of each epoch, in order to resume training in case of failure.
                    This directory will be emptied at the end of training.

                    .. dropdown:: **epoch-15**
                        :icon: file-directory
                        :color: muted

                        Checkpoint at the end of epoch ``15``.

                        .. dropdown:: model.pth.tar → ``maps.training.splits[0].tmp.epochs[15].model``
                            :icon: file
                            :color: light

                            The weights of the model.

                        .. dropdown:: scaler.json → ``maps.training.splits[0].tmp.epochs[15].scaler``
                            :icon: file
                            :color: light

                            The state of the `Gradient Scaler <https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling>`_
                            at epoch ``15``.

                        .. dropdown:: state.json → ``maps.training.splits[0].tmp.epochs[15].state``
                            :icon: file
                            :color: light

                            The :py:class:`state of the Trainer <clinicadl.train.TrainerState>` at epoch ``15``.

                        .. dropdown:: **callbacks** → maps.training.splits[0].tmp.epochs[15].callbacks
                            :icon: file-directory
                            :color: muted

                            Where :py:mod:`~clinicadl.callbacks`' checkpoints are saved.

                        .. dropdown:: **validation_metrics**
                            :icon: file-directory
                            :color: muted

                            Validation metrics at epoch ``15``.

                            .. dropdown:: aggregated.tsv → ``maps.training.splits[0].tmp.epochs[15].validation_metrics.aggregated``
                                :icon: file
                                :color: light

                                Aggregated validation metrics.

                            .. dropdown:: details.tsv → ``maps.training.splits[0].tmp.epochs[15].validation_metrics.details``
                                :icon: file
                                :color: light

                                Validation metrics for each image.

                .. dropdown:: **validation_metrics**
                    :icon: file-directory
                    :color: muted

                    Validation metrics at every validation step.

                    .. dropdown:: aggregated.tsv → ``maps.training.splits[0].validation_metrics.aggregated``
                        :icon: file
                        :color: light

                        Aggregated validation metrics.

                    .. dropdown:: details.tsv → ``maps.training.splits[0].validation_metrics.details``
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

                .. dropdown:: dataset.json → ``maps.test.groups["X"].dataset_json``
                    :icon: file
                    :color: light

                    Details on the :py:class:`dataset <clinicadl.data.datasets>` used
                    for the group ``"X"``.

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

                        .. dropdown:: **metrics**
                            :icon: file-directory
                            :color: muted

                            Metrics on the group ``"X"``.

                            .. dropdown:: aggregated.tsv → ``maps.test.groups["X"].splits[0].models["epoch-10"].metrics.aggregated``
                                :icon: file
                                :color: light

                                Aggregated metrics.

                            .. dropdown:: details.tsv → ``maps.test.groups["X"].splits[0].models["epoch-10"].metrics.details``
                                :icon: file
                                :color: light

                                Metrics for each image.


                    .. dropdown:: **final**
                        :icon: file-directory
                        :color: muted

                        Results obtained with the final model trained on the split ``0``
                        (i.e. the model in ``maps_dir/training/split-0/models/final``).

                        .. dropdown:: **metrics**
                            :icon: file-directory
                            :color: muted

                            Metrics on the group ``"X"``.

                            .. dropdown:: aggregated.tsv → ``maps.test.groups["X"].splits[0].models["final"].metrics.aggregated``
                                :icon: file
                                :color: light

                                Aggregated metrics.

                            .. dropdown:: details.tsv → ``maps.test.groups["X"].splits[0].models["final"].metrics.details``
                                :icon: file
                                :color: light

                                Metrics for each image.

                    .. dropdown:: **best-mse**
                        :icon: file-directory
                        :color: muted

                        Results obtained with the best model with respect to the metric ``"mse"``, trained on the split ``0``
                        (i.e. the model in ``maps_dir/training/split-0/models/best_models/best-mse``).

                        .. dropdown:: **metrics**
                            :icon: file-directory
                            :color: muted

                            Metrics on the group ``"X"``.

                            .. dropdown:: aggregated.tsv → ``maps.test.groups["X"].splits[0].models["best-mse"].metrics.aggregated``
                                :icon: file
                                :color: light

                                Aggregated metrics.

                            .. dropdown:: details.tsv → ``maps.test.groups["X"].splits[0].models["best-mse"].metrics.details``
                                :icon: file
                                :color: light

                                Metrics for each image.

        .. dropdown:: **prediction**
            :icon: file-directory
            :color: muted

            Results of :py:meth:`Trainer.predict <clinicadl.train.Trainer.predict>`.

            .. dropdown:: **group-X**
                :icon: file-directory
                :color: muted

                Inferences on the group ``"X"``.

                .. dropdown:: data.tsv → ``maps.prediction.groups["X"].data_tsv``
                    :icon: file
                    :color: light

                    List of all (participant, session) pairs in the group ``"X"``.

                .. dropdown:: dataset.json → ``maps.prediction.groups["X"].dataset_json``
                    :icon: file
                    :color: light

                    Details on the :py:class:`dataset <clinicadl.data.datasets>` used
                    for the group ``"X"``.

                .. dropdown:: **split-0**
                    :icon: file-directory
                    :color: muted

                    Inferences on the group ``"X"`` obtained with models trained on split ``0``
                    (i.e. the models in ``maps_dir/training/split-0/models``).

                    .. dropdown:: **epoch-10**
                        :icon: file-directory
                        :color: muted

                        Inferences obtained with the model at epoch ``10``, trained on split ``0``
                        (i.e. the model in ``maps_dir/training/split-0/models/checkpoints/epoch-10``).

                        .. dropdown:: output.tsv → ``maps.prediction.groups["X"].splits[0].models["epoch-10"].output_tsv``
                            :icon: file
                            :color: light

                            If outputs of the model are scalars, they will be stored in this DataFrame.

                        .. dropdown:: **caps_output** → maps.prediction.groups["X"].splits[0].models["epoch-10"].caps_output
                            :icon: file-directory
                            :color: muted

                            If outputs of the model are images, they will be stored in a :term:`CAPS` directory here.


                    .. dropdown:: **final**
                        :icon: file-directory
                        :color: muted

                        Inferences obtained with the final model trained on the split ``0``
                        (i.e. the model in ``maps_dir/training/split-0/models/final``).

                        .. dropdown:: output.tsv → ``maps.prediction.groups["X"].splits[0].models["final"].output_tsv``
                            :icon: file
                            :color: light

                            If outputs of the model are scalars, they will be stored in this DataFrame.

                        .. dropdown:: **caps_output** → maps.prediction.groups["X"].splits[0].models["final"].caps_output
                            :icon: file-directory
                            :color: muted

                            If outputs of the model are images, they will be stored in a :term:`CAPS` directory here.

                    .. dropdown:: **best-mse**
                        :icon: file-directory
                        :color: muted

                        Inferences obtained with the best model with respect to the metric ``"mse"``, trained on the split ``0``
                        (i.e. the model in ``maps_dir/training/split-0/models/best_models/best-mse``).

                        .. dropdown:: output.tsv → ``maps.prediction.groups["X"].splits[0].models["best-mse"].output_tsv``
                            :icon: file
                            :color: light

                            If outputs of the model are scalars, they will be stored in this DataFrame.

                        .. dropdown:: **caps_output** → maps.prediction.groups["X"].splits[0].models["best-mse"].caps_output
                            :icon: file-directory
                            :color: muted

                            If outputs of the model are images, they will be stored in a :term:`CAPS` directory here.


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

        >>> maps.training.splits[0].models.checkpoints.epochs[10].model
        PosixPath('maps_dir/training/split-0/models/checkpoints/epoch-10/model.pth.tar')

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

    @property
    def training(self) -> TrainingDir:
        """Directory containing the information on trainings."""
        return self._training

    @property
    def test(self) -> TestDir:
        """Directory containing the information of tests."""
        return self._test

    @property
    def prediction(self) -> PredictionDir:
        """Directory containing the information of predictions."""
        return self._prediction

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
    def nn_summary_txt(self) -> Path:
        return (self.path / f"{NN}_{SUMMARY}").with_suffix(TXT)

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
    def load_file(path: PathType) -> Any:
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

            from clinicadl.io import Maps

            maps = Maps("maps_dir")
            maps.read()

        .. code-block:: python

            >>> maps.load_file(maps.metrics_json)
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
            >>> maps.load_file(maps.training.splits[0].models.checkpoints.epochs[1].model)
            OrderedDict([('conv0.weight',
              tensor([[[[ 7.2531e-03,  5.7384e-03,  1.4988e-02,  ..., -3.1380e-02,
                         -2.2103e-02,  5.8432e-02],
            ...
            >>> maps.load_file(maps.training.data.data_tsv)
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
        elif path.suffix == LOG or path.suffix == TXT:
            with path.open("r", encoding="utf-8") as f:
                return f.read()
        elif path.suffix == TSV:
            return pd.read_csv(path, sep=SEP)
        elif path.suffix == TAR and path.with_suffix("").suffix == PTH:
            return torch_load(path)

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
            directory are ``.json``, ``.txt``, ``.log``, ``.tsv``, and ``.pth.tar``.

        Examples
        --------

        .. code-block:: python

            from clinicadl.io import Maps

            maps = Maps("maps_dir")
            maps.read()

        .. code-block:: python

            >>> metrics = {
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
            >>> maps.save_file(metrics, maps.metrics_json)

        See Also
        --------
        :py:meth:`load_file`

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
            assert isinstance(obj, pd.DataFrame)
            obj.to_csv(path, sep=SEP, index=False)
        elif path.suffix == TAR and path.with_suffix("").suffix == PTH:
            torch_save(obj, path)
        else:
            raise ValueError(
                f"'{path.suffix}' files are not supported. The supported files in a MAPS directory are {[JSON, LOG, TXT, TSV, PTH + TAR]}"
            )

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
