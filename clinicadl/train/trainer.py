from __future__ import annotations

import warnings
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Generator,
    Iterator,
    Optional,
    TypeVar,
    Union,
)

import torch
from torch.amp.autocast_mode import autocast
from typing_extensions import Self

from clinicadl.callbacks import CallbacksHandler
from clinicadl.callbacks.base import Event
from clinicadl.data.dataloader import Batch, DataLoaderConfig
from clinicadl.data.datasets.factory import get_dataset_from_json
from clinicadl.io.maps.maps import Maps
from clinicadl.metrics import MetricsHandler
from clinicadl.metrics.config import LossMetricConfig
from clinicadl.models.factory import get_model_from_json
from clinicadl.optim.config import OptimizationConfig
from clinicadl.split.split import Split
from clinicadl.train.computational import ComputationalConfig
from clinicadl.train.trainer_state import TrainerState
from clinicadl.utils.dictionary.words import CPU
from clinicadl.utils.enum import TrainerCall
from clinicadl.utils.exceptions import CannotReadJsonError, CannotReadJsonFieldError
from clinicadl.utils.seed import seed_everything_context

if TYPE_CHECKING:
    from torch.amp.grad_scaler import GradScaler
    from torch.optim import Optimizer

    from clinicadl.callbacks import Callback
    from clinicadl.data.dataloader import BatchType, DataLoader
    from clinicadl.data.datasets import Dataset
    from clinicadl.io.maps.inference import InferenceDirType
    from clinicadl.io.maps.utils import DataDir
    from clinicadl.metrics.types import MetricOrConfig
    from clinicadl.models import Model
    from clinicadl.utils.typing import PathType


T = TypeVar("T")


class Trainer:
    """
    The core class to manage model **training**, **evaluation**, and **prediction**.

    This class makes all ``ClinicaDL`` objects work together in order to have
    functional training, evaluation,  and prediction pipelines. It hides the logic
    common to most PyTorch workflow, while allowing the user to customize the essential
    logic in the :py:class:`~clinicadl.models.Model`, or the non-essential logic via
    :py:class:`callbacks clinicadl.callbacks.Callbacks`.

    ``Trainer`` also offers high-performance computing options, configured via
    :py:class:`clinicadl.train.ComputationalConfig`.

    ``Trainer`` will record all outputs and results in a :term:`Maps`
    directory, as well as the configurations used in order to reproduce the experiment.

    Main methods:
    - :py:meth:`train`: to train your model;
    - :py:meth:`resume`: to resume an interrupted training;
    - :py:meth:`validate`: to compute new metrics on your validation data;
    - :py:meth:`test`: to evaluate your model on test data.

    Parameters
    ----------
    maps : Union[PathType, Maps]
        Directory where outputs, results and configurations will be saved.

    model : Model
        The :py:class:`clinicadl.models.Model` to train or evaluate.

    metrics : Union[dict[str, MetricOrConfig], MetricsHandler], default={"loss": LossMetricConfig(loss_name="loss")}
        Dictionary of metric names and metric instances for monitoring model performance.
        Metric instances can be passed via a :py:class:`clinicadl.metrics.config.MetricsConfig` or
        a :py:class:`clinicadl.metrics.Metric`.\n
        A :py:class:`~clinicadl.metrics.MetricsHandler` containing the metrics can also be passed directly.

        .. note::
            - Here you define the metrics useful within the scope of your ``Trainer``,
             but it doesn't mean that they will be all always computed. The list of the metrics to compute
             will be given to the method of the trainer.
            - You can still add metrics later via :py:meth:`add_metrics`.

        By default, only the loss returned by :py:meth:`Model.get_loss_functions <clinicadl.models.Model.get_loss_functions>`
        will be monitored (it is expected to be called ``"loss"``).

    optimization : OptimizationConfig, default=OptimizationConfig()
        Configuration for the optimization of the neural network during training
        (e.g. the number of epochs), passed via a :py:class:`~clinicadl.optim.OptimizationConfig`.

    callbacks : Optional[Union[list[Callback], CallbacksHandler]], default=None
        List of :py:class:`~clinicadl.callbacks.Callback` to customize the trainer.
        A :py:class:`~clinicadl.callbacks.CallbacksHandler` containing the desired callbacks can also be passed directly.\n
        If ``None``, only the default callbacks in :py:class:`~clinicadl.callbacks.CallbacksHandler` will
        be applied.\n

    overwrite : bool, default=False
        Whether to overwrite the :term:`MAPS` if it exists.

    """

    def __init__(
        self,
        maps: Union[PathType, Maps],
        model: Model,
        metrics: Union[dict[str, MetricOrConfig], MetricsHandler] = {
            "loss": LossMetricConfig(loss_name="loss")
        },
        optimization: OptimizationConfig = OptimizationConfig(),
        callbacks: Optional[Union[list[Callback], CallbacksHandler]] = None,
        overwrite: bool = False,
    ) -> None:
        if not isinstance(maps, Maps):
            maps = Maps(maps)
        maps.create(overwrite=overwrite)

        self._instantiate_attributes(maps, model, metrics, optimization, callbacks)

    @property
    def maps(self) -> Maps:
        """
        The py:class:`clinicadl.io.Maps` associated to the ``Trainer``.
        """
        return self._maps

    @property
    def model(self) -> Model:
        """
        The py:class:`clinicadl.models.Model` associated to the ``Trainer``.
        """
        return self._model

    @property
    def metrics(self) -> MetricsHandler:
        """
        The py:class:`metrics clinicadl.metrics` computed by ``Trainer``.
        """
        return self._metrics

    @property
    def callbacks(self) -> CallbacksHandler:
        """
        The py:class:`callbacks clinicadl.callbacks` called by ``Trainer``.
        """
        return self._callbacks

    @property
    def optimization(self) -> OptimizationConfig:
        """
        The py:class:`optimization configuration clinicadl.optim.OptimizationConfig`
        of the ``Trainer``.
        """
        return self._optim_config

    @property
    def state(self) -> TrainerState:
        """
        The current py:class:`state clinicadl.state.TrainerState` of the ``Trainer``.
        """
        return self._state

    def _instantiate_attributes(
        self,
        maps: Maps,
        model: Model,
        metrics: Union[dict[str, MetricOrConfig], MetricsHandler],
        optimization: OptimizationConfig = OptimizationConfig(),
        callbacks: Optional[Union[list[Callback], CallbacksHandler]] = None,
    ) -> None:
        """
        Defines the Trainer's attributes.
        """
        self._maps = maps
        self._model = model

        self._initial_state_dict = None
        if not optimization.reset_model:
            self._initial_state_dict = deepcopy(self._model.cpu().state_dict())

        if isinstance(metrics, MetricsHandler):
            self._metrics = metrics
        else:
            self._metrics = MetricsHandler(**metrics)
        self._metrics.init_metrics(self._model)

        if isinstance(callbacks, CallbacksHandler):
            self._callbacks = callbacks
        else:
            self._callbacks = CallbacksHandler(callbacks=callbacks if callbacks else [])

        self._optim_config = optimization

        self._state = TrainerState()

        self._call_event(
            Event.INIT,
            metrics=self.metrics,
            optimization=self.optimization,
            callbacks=self.callbacks,
        )

    @classmethod
    def from_maps(cls, maps_path: PathType, **kwargs) -> Self:
        """
        To restore a ``Trainer`` from a :term:`MAPS` directory.

        This classmethod will try to restore the ``Trainer`` that
        created this :term:`MAPS` by reading the ``json`` configuration
        files saved inside.

        If errors happen when restoring a component of the ``Trainer``
        (e.g. you use your own :py:class:`~clinicadl.models.Model`, so the ``Trainer`` cannot read it),
        you can restore the associated objects on your own and pass them via keyword arguments
        (e.g. ``Trainer.train(..., model=...))``.

        Parameters
        ----------
        maps_path : PathType
            Path to the :term:`MAPS` directory.
        **kwargs : Any
            To pass objects that the ``Trainer`` cannot restore.

        Returns
        -------
        Self
            The ``Trainer`` associated to the :term:`MAPS`.

        Examples
        --------
        .. code-block::
            trainer = Trainer(
            maps="maps",
            model=MyModel()
            ...
            )
        ...

        .. code-block::
            >>> Trainer.from_maps("maps")
            CannotReadJsonError: Cannot read the model [...]    # if MyModel is not a model supported natively by ClinicaDL
            >>> Trainer.from_maps("maps", model=MyModel())
        """
        maps = Maps(maps_path)
        maps.read()

        model = cls._read_attribute_in_json(
            "model", maps.model_json, get_model_from_json, kwargs
        )
        metrics = cls._read_attribute_in_json(
            "metrics", maps.metrics_json, MetricsHandler.from_json, kwargs
        )
        callbacks = cls._read_attribute_in_json(
            "callbacks", maps.callbacks_json, CallbacksHandler.from_json, kwargs
        )
        optimization = OptimizationConfig.from_json(maps.training.optimization_json)

        trainer = cls.__new__(cls)  # bypass init because maps exist
        trainer._instantiate_attributes(maps, model, metrics, optimization, callbacks)

        return trainer

    @staticmethod
    def _read_attribute_in_json(
        name: str, path: Path, reader: Callable[[Path], T], kwargs: dict[str, Any]
    ) -> T:
        """
        Reads a Trainer attribute in a json, and handles potential reading errors.
        """
        try:
            return kwargs.get(name) or reader(path)
        except CannotReadJsonFieldError as e:
            raise CannotReadJsonError(
                f"Cannot read the {name} (in {path}). Please pass it to from_maps via a keyword "
                f"argument (e.g. Trainer.from_maps(..., {name}=...))."
            ) from e

    def add_metrics(self, **metrics: MetricOrConfig) -> None:
        """
        To add new metrics to compute to the `py:class:`clinicadl.metrics.MetricsHandler`.

        Parameters
        ----------
        **metrics : MetricConfig
            The metrics, passed as a :py:class:`clinicadl.metrics.config.MetricConfig`
            or a :py:class:`clinicadl.metrics.Metric`.
        """
        self.metrics.add_metrics(**metrics)
        self.metrics.to_json(self._maps.metrics_json, overwrite=True)

    def add_callbacks(self, callbacks: Sequence[Callback]) -> None:
        """
        To add new callbacks to the `py:class:`clinicadl.callbacks.CallbacksHandler`.

        Parameters
        ----------
        callbacks : Sequence[Callback]
            The :py:class:`Callbacks <clinicadl.callbacks.Callback` to add.
        """
        self._callbacks.add_callbacks(callbacks)
        self._callbacks.to_json(self._maps.callbacks_json, overwrite=True)

    def train(
        self,
        split: Split,
        computational: Optional[ComputationalConfig] = None,
        metrics: Optional[Sequence[str]] = None,
    ) -> None:
        """
        To train a model.

        Parameters
        ----------
        split : Split
            The :py:class:`clinicadl.split.Split` containing the training and validation data.

        computational : Optional[ComputationalConfig], default=None
            Computational configuration, passed via a :py:class:`clinicadl.train.ComputationalConfig`.
            This is where you can setup high-performance computing features or a seed to make
            your training reproducible.\n

            - If ``resume=True``, the user **cannot pass a computational configuration**, because the previous
              one will be used.

            - If ``resume=False`` and ``computational=None``, the default parameters defined in
              :py:class:`~clinicadl.train.ComputationalConfig` will be used.

        metrics : Optional[Sequence[str]], default=None
            The names of the metric to compute on the validation data. The metrics mentioned
            here must have been defined beforehand when instantiating the ``Trainer`` or via
            :py:meth:`add_metrics`.\n
            By default, all the defined metrics will be computed.

        See Also
        --------
        - :py:meth:`resume`
            To resume an interrupted training.
        """
        self.maps.read()
        self._create_split(split.index)

        with self._seed_context(
            computational.seed, computational.deterministic
        ), self._exception_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._train(
                split=split, computational=computational, metrics=metrics, resume=False
            )

    def resume(self, split_idx: int, split: Optional[Split] = None) -> None:
        """
        To resume an interrupted training launched with :py:meth:`train`.

        ``Trainer`` will look for the last checkpoint saved with :py:class:`clinicadl.callbacks.TrainingCheckpointCallback`,
        and restart training from there.

        ``Trainer`` will attempt to load your training and validation data from the :term:`MAPS`
        directory, so typically providing the split index is sufficient. However, it it fails, you can
        manually supply the split.

        .. important::
            Here, the computational setup will be the same as the one used when training
            the model before the interruption. So, if the model was first trained on a GPU,
            make sure a GPU is available when calling ``validate``.

        Parameters
        ----------
        split_idx : int
            The index of the split to resume training on.

        dataloader : Optional[Split], default=None
            The :py:class:`clinicadl.split.Split` containing the training and validation data.\n

            If you pass a split here, it will be used; otherwise, ``Trainer``
            will attempt to load the  split from the :term:`MAPS`.

        """
        self.maps.read()
        self._check_split_exists(split_idx)

        if not split:
            split = self._get_split(split_idx)
        else:
            assert (
                split_idx == split.index
            ), f"split_idx does not match split.index. Got {split_idx} and {split.index}"

        computational = self._get_old_computational_config(split_idx)

        with self._seed_context(
            computational.seed, computational.deterministic
        ), self._exception_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._train(
                split=split,
                computational=computational,
                metrics=None,  # metrics filtering is done when loading the checkpoint
                resume=True,
            )

    def _train(
        self,
        split: Split,
        computational: ComputationalConfig,
        metrics: Optional[Sequence[str]],
        resume: bool,
    ) -> None:
        """
        Instantiates/restarts the required objects (e.g. optimizers),
        sends the model to the specified device and starts the
        training loop.
        """
        if metrics:
            metrics_handler = self.metrics.subset(metrics)
        else:
            metrics_handler = self.metrics

        optimizers = self.model.build_optimizers()
        grad_scaler = computational.get_scaler()

        self._reset_train(
            split=split,
            metrics=metrics_handler,
            optimizers=optimizers,
        )
        self._model_to(computational)

        if resume:
            self._call_event(
                Event.RESUME,
                split=split,
                optimizers=optimizers,
                grad_scaler=grad_scaler,
                optimization=self.optimization,
                metrics=metrics_handler,
                callbacks=self.callbacks,
                computational=computational,
            )
        else:
            self._call_event(
                Event.TRAIN_START,
                split=split,
                optimizers=optimizers,
                optimization=self.optimization,
                metrics=metrics_handler,
                callbacks=self.callbacks,
                computational=computational,
            )

        self._train_loop(
            split=split,
            optimizers=optimizers,
            grad_scaler=grad_scaler,
            metrics=metrics_handler,
            computational=computational,
        )

        self._call_event(Event.TRAIN_END)

    def _train_loop(
        self,
        split: Split,
        optimizers: dict[str, Optimizer],
        grad_scaler: GradScaler,
        metrics: MetricsHandler,
        computational: ComputationalConfig,
    ) -> None:
        """
        The core training logic with the iteration on the epochs, and the
        nested iteration on the training batches.
        """
        for epoch in range(
            self.state.current_epoch + 1, self.optimization.num_epochs + 1
        ):
            if self.state.should_stop:
                break

            self._reset_epoch(epoch, train_loader=split.train_loader)

            self._call_event(Event.EPOCH_START)

            for batch_idx, batch in enumerate(split.train_loader, start=1):
                self.state.current_train_batch = batch_idx

                self._call_event(Event.BATCH_START, batch=batch)

                self._batch_to(batch, computational=computational)

                self._call_event(Event.FORWARD_START, batch=batch)

                with autocast(
                    device_type=computational.device.type,
                    enabled=computational.amp,
                ):
                    loss = self.model.forward_step(batch)

                self._call_event(
                    Event.BACKWARD_START, loss=loss, grad_scaler=grad_scaler
                )

                self.model.backward_step(loss, grad_scaler=grad_scaler)

                self._call_event(Event.BACKWARD_END)

                if batch_idx % self.optimization.accumulation_steps == 0:
                    self._call_event(
                        Event.OPTIM_STEP_START,
                        optimizers=optimizers,
                        grad_scaler=grad_scaler,
                    )

                    self._clip_gradients()

                    self.model.optimization_step(
                        optimizers=optimizers, grad_scaler=grad_scaler
                    )
                    self.state.optim_step += 1

                    grad_scaler.update()
                    for optimizer in optimizers.values():
                        optimizer.zero_grad(set_to_none=True)

                    self._call_event(
                        Event.OPTIM_STEP_END,
                        optimizers=optimizers,
                        grad_scaler=grad_scaler,
                    )

                self._call_event(Event.BATCH_END)

            if (
                self.state.current_epoch
                % self.optimization.evaluation_interval  # always validate the first epoch
                == 0
            ):
                self._validation(
                    split.val_loader,
                    metrics=metrics,
                    computational=computational,
                )

            self._call_event(Event.EPOCH_END)

    def validate(
        self,
        split_idx: int,
        metrics: Sequence[str],
        dataloader: Optional[DataLoader] = None,
        model_checkpoint: Optional[str] = None,
    ) -> None:
        """
        To evaluate your model on your validation data with new metrics.

        This method should be used when you wish to compute new validation
        metrics once the training phase is complete.

        ``Trainer`` attempts to load your validation data from the :term:`MAPS`
        directory, so typically providing the split index is sufficient. However, it it fails, you can
        manually supply the validation dataloader.

        .. important::
            Here, the computational setup will be the same as the one used when training
            the model. So, if the model was trained on a GPU, make sure a GPU is available when
            calling ``validate``.

        Parameters
        ----------
        split_idx : int
            The index of the split on which the model to validate has been trained.

        metrics : Sequence[str]
            The names of the metric to compute on the validation data. The metrics mentioned
            here must have been defined beforehand when instantiating the ``Trainer`` or via
            :py:meth:`add_metrics`.

        dataloader : Optional[DataLoader], default=None
            The dataloader containing the validation data on which the model will be evaluated.
            If you pass a dataloader, it will be your validation data, otherwise, ``Trainer```
            will attempt to load the validation data from the :term:`MAPS`.

            .. important::
                ``dataloader`` must be an output of :py:meth:`clinicadl.data.DataLoaderConfig.get_object`.
                Not just any PyTorch's dataloader is accepted.

        model_checkpoint: Optional[str], default=None
            The name of the model checkpoint to validate. If ``None``, all the checkpoints will
            be evaluated. The name of the checkpoint must follow one of these formats:

            - ``"best-<metric-name>"``: the best model trained on split ``split_idx``
              w.r.t. the metric ``<metric-name>``;
            - ``"epoch-<epoch-idx>"``: the checkpoint of the model trained on split ``split_idx``
              at epoch ``<epoch-idx>``;
            - ``"final"``: the model at the end of training on split ``split_idx``.

        Examples
        --------
        .. code-block::
            ...
            from clinicadl.train import Trainer
            from clinicadl.metrics.config import (
                LossMetricConfig,
                DiceMetricConfig,
                SurfaceDiceMetricConfig,
            )
            from clinicadl.callbacks import ModelCheckpointCallback

            trainer = Trainer(
                ...
                metrics={"loss": LossMetricConfig(), "dice": DiceMetricConfig()},
                callbacks=[ModelCheckpointCallback(metric="loss")],  # save the best model w.r.t. "loss"
            )

            trainer.train(...)  # training on split 1, validation on "loss" and "dice"

            trainer.add_metrics(
                nsd=SurfaceDiceMetricConfig(class_thresholds=[1])
            )  # define a new metric named "nsd"

        .. code-block::
            trainer.validate(
                split_idx=1, metrics=["nsd"], model_checkpoint="best-loss"
            )  # validation with metric "nsd"

        .. code-block::
            trainer.validate(split_idx=1, metrics=["nsd"])  # validation of all checkpoints

        .. code-block::
            >>> trainer.validate(split_idx=1, metrics=["nsd"])
            CannotReadJsonError: ClinicaDL could not read the dataloader in ...
            >>> trainer.validate(
                    split_idx=1, metrics=["nsd"], dataloader=dataloader
                )  # pass the dataloader if a reading error is raised
        """
        self.maps.read()
        self._check_split_exists(split_idx)

        computational = self._get_old_computational_config(split_idx)

        with self._seed_context(
            computational.seed, computational.deterministic
        ), self._exception_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._validate(
                split_idx=split_idx,
                metrics=metrics,
                dataloader=dataloader,
                model_checkpoint=model_checkpoint,
                computational=computational,
            )

    def _validate(
        self,
        split_idx: int,
        metrics: Sequence[str],
        dataloader: Optional[DataLoader] = None,
        model_checkpoint: Optional[str] = None,
        computational: ComputationalConfig = ComputationalConfig(),
    ) -> None:
        """
        Instantiates/restarts the required objects (e.g. metrics),
        loads the model weights, sends the model to the specified device and starts the
        evaluation loop.
        """
        if not dataloader:
            dataloader = self._get_dataloader(
                self.maps.training.data.validation.splits[split_idx]
            )

        metrics_handler = self.metrics.subset(metrics)

        for chkpt_name, chkpt in self._get_models_in_split(split_idx, model_checkpoint):
            self._reset_validate(split_idx, dataloader, metrics_handler)
            self._load_model_checkpoint(chkpt)
            self._model_to(computational)

            self._call_event(
                Event.VALIDATE_START,
                dataloader=dataloader,
                model_checkpoint=chkpt_name,
                metrics=metrics_handler,
                callbacks=self.callbacks,
                computational=computational,
            )

            self._evaluation_loop(
                dataloader,
                metrics=metrics_handler,
                computational=computational,
            )

            self._call_event(
                Event.VALIDATE_END,
                metrics=metrics_handler,
            )

    def test(
        self,
        model_checkpoint: str,
        group_name: str,
        metrics: Optional[Sequence[str]] = None,
        dataloader: Optional[DataLoader] = None,
        computational: ComputationalConfig = ComputationalConfig(),
    ) -> None:
        """
        To test you model.

        This method checks for :term:`data leakage` between your test and your training/validation data.

        Your test data must be associated with a ``group_name``. If the ``group_name`` already exists
        in the :term:`MAPS`, ``Trainer`` attempts to load the test data from it. However, it it fails, you can
        manually supply the test dataloader.

        Parameters
        ----------
        model_checkpoint : str
            The name of the model checkpoint to test. The name of the checkpoint must follow one of these formats:

            - ``"split-<split-idx>_best-<metric-name>"``: the best model trained on split ``<split-idx>``
              w.r.t. the metric ``<metric-name>``;
            - ``"split-<split-idx>_epoch-<epoch-idx>"``: the checkpoint of the model trained on split ``<split-idx>``
              at epoch ``<epoch-idx>``;
            - ``"split-<split-idx>_final"``: the model at the end of training on split ``<split-idx>``.

        group_name : str
            The name given to these test data. If the group name was already used, ``Trainer``
            will attempt to restore the data from the :term:`MAPS`; otherwise, you must pass
            a dataloader containing your test data.

        metrics : Optional[Sequence[str]], default=None
            The names of the metric to compute on the test data. The metrics mentioned
            here must have been defined beforehand when instantiating the ``Trainer`` or via
            :py:meth:`add_metrics`.\n
            By default, all the defined metrics will be computed.

        dataloader : Optional[DataLoader], default=None
            The dataloader containing the test data on which the model will be evaluated.
            If you pass a dataloader, it will be your test data; otherwise, ``Trainer```
            will attempt to load the test data from the :term:`MAPS`.

            .. important::
                ``dataloader`` must be an output of :py:meth:`clinicadl.data.DataLoaderConfig.get_object`.
                Not just any PyTorch's dataloader is accepted.

        computational : ComputationalConfig, default=ComputationalConfig()
            Computational configuration, passed via a :py:class:`clinicadl.train.ComputationalConfig`.
            This is where you can setup high-performance computing features.

        Examples
        --------
        .. code-block::
            ...
            from clinicadl.train import Trainer
            from clinicadl.metrics.config import (
                LossMetricConfig,
                DiceMetricConfig,
            )
            from clinicadl.callbacks import ModelCheckpointCallback

            trainer = Trainer(
                ...
                metrics={"loss": LossMetricConfig(), "dice": DiceMetricConfig()},
                callbacks=[
                    ModelCheckpointCallback(metric="loss")
                ],  # save the best model w.r.t. "loss"
            )

            trainer.train(...)  # training on split 1

        .. code-block::
            trainer.test(
                model_checkpoint="split-1_best-loss", group_name="adni"
            )  # compute all the metrics

        .. code-block::
            trainer.test(
                model_checkpoint="split-1_best-loss", group_name="adni", metrics=["dice"]
            )  # compute only "dice"

        .. code-block::
            >>> trainer.test(model_checkpoint="split-1_best-loss", group_name="adni")
            CannotReadJsonError: ClinicaDL could not read the dataloader in ...
            >>> trainer.test(
                model_checkpoint="split-1_best-loss", group_name="adni", dataloader=dataloader
            )  # pass the dataloader if the Trainer cannot read the dataloader associated to "adni"
        """
        with self._seed_context(
            computational.seed, computational.deterministic
        ), self._exception_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._test(
                model_checkpoint=model_checkpoint,
                metrics=metrics,
                group_name=group_name,
                dataloader=dataloader,
                computational=computational,
            )

    def _test(
        self,
        model_checkpoint: str,
        group_name: str,
        metrics: Optional[Sequence[str]],
        dataloader: Optional[DataLoader] = None,
        computational: ComputationalConfig = ComputationalConfig(),
    ) -> None:
        """
        Instantiates/restarts the required objects (e.g. metrics),
        loads the model weights, sends the model to the specified device and starts the
        evaluation loop.
        """
        self.maps.read()

        if not dataloader:
            self._check_group_exists(group_name)
            dataloader = self._get_dataloader(self.maps.test.groups[group_name])

        self._create_results_dir(
            self.maps.test, group_name=group_name, model_checkpoint=model_checkpoint
        )

        if metrics:
            metrics_handler = self.metrics.subset(metrics)
        else:
            metrics_handler = self.metrics

        self._reset_test(dataloader, metrics_handler)
        self._load_model_checkpoint(
            self.maps.training.get_checkpoint_dir(model_checkpoint).model_pt
        )
        self._model_to(computational)

        self._call_event(
            Event.TEST_START,
            dataloader=dataloader,
            model_checkpoint=model_checkpoint,
            metrics=metrics_handler,
            group_name=group_name,
            callbacks=self.callbacks,
            computational=computational,
        )

        self._evaluation_loop(
            dataloader, metrics=metrics_handler, computational=computational
        )

        self._call_event(
            Event.TEST_END,
            metrics=metrics_handler,
        )

    def predict(
        self,
        model_checkpoint: str,
        group_name: str,
        dataloader: Optional[DataLoader] = None,
        computational: ComputationalConfig = ComputationalConfig(),
    ) -> None:
        """
        .. admonition:: Not Implemented
            :class: warning

            ``Trainer.predict`` will be implemented in a future release.
        """
        raise NotImplementedError(
            "Trainer.predict will be implemented in a future release"
        )

    def _validation(
        self,
        dataloader: DataLoader,
        metrics: MetricsHandler,
        computational: ComputationalConfig,
    ) -> None:
        """
        Validation phase during training.
        """
        self._reset_validation(dataloader, metrics)

        self._call_event(
            Event.VAL_START,
            dataloader=dataloader,
            metrics=metrics,
        )

        self._evaluation_loop(
            dataloader,
            metrics=metrics,
            epoch=self.state.current_epoch,
            computational=computational,
        )

        self._call_event(
            Event.VAL_END,
            metrics=metrics,
        )

    def _evaluation_loop(
        self,
        dataloader: DataLoader,
        metrics: MetricsHandler,
        computational: ComputationalConfig,
        epoch: Optional[int] = None,
    ) -> None:
        """
        The core evaluation logic, with the iteration on the
        evaluation batches.
        """
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader, start=1):
                if self.state.called == TrainerCall.TEST:
                    self.state.current_test_batch = batch_idx
                else:
                    self.state.current_val_batch = batch_idx

                self._call_event(Event.BATCH_START, batch=batch)

                self._batch_to(batch, computational=computational)

                self._call_event(Event.EVAL_START, batch=batch)

                with autocast(
                    device_type=computational.device.type,
                    enabled=computational.amp,
                ):
                    output_batch = self.model.evaluation_step(batch)

                self._call_event(
                    Event.METRIC_START,
                    output=output_batch,
                    metrics=metrics,
                )

                metrics_df = metrics(output_batch, epoch=epoch)

                self._call_event(
                    Event.METRIC_END,
                    detailed_metrics_df=metrics_df,
                )

                self._call_event(Event.BATCH_END)

        metrics.aggregate(epoch=epoch)

    def _reset_train(
        self, split: Split, metrics: MetricsHandler, optimizers: dict[str, Optimizer]
    ) -> None:
        """
        Resets the relevant objects before training.
        """
        self.state.reset_training(
            split_idx=split.index, num_epochs=self._optim_config.num_epochs
        )
        self._reset_model()
        split.train_dataset.train()
        split.val_dataset.eval()
        metrics.reset(reset_df=True)
        for optimizer in optimizers.values():
            optimizer.zero_grad()

    def _reset_model(self):
        """
        Resets the model according to the resetting strategy.
        """
        self.model.to(
            device=CPU, memory_format=torch.contiguous_format
        )  # otherwise seeding will not be consistent across memory formats and devices!
        if self.optimization.reset_model:
            self.model.reset()
        else:
            self.model.load_state_dict(self._initial_state_dict)

    def _reset_epoch(self, epoch: int, train_loader: DataLoader) -> None:
        """
        Resets the relevant objects before a new epoch.
        """
        self.state.reset_epoch(current_epoch=epoch, train_loader=train_loader)
        self.model.train()
        train_loader.set_epoch(epoch)

    def _reset_validation(
        self, val_loader: DataLoader, metrics: MetricsHandler
    ) -> None:
        """
        Resets the relevant objects before a validation phase in :py:meth:`train`.
        """
        self.state.reset_validation(
            split_idx=self.state.split_idx, val_loader=val_loader, in_training=True
        )
        self.model.eval()
        metrics.reset(reset_df=False)

    def _reset_validate(
        self, split_idx: int, dataloader: DataLoader, metrics: MetricsHandler
    ) -> None:
        """
        Resets the relevant objects before a validation phase in :py:meth:`validate`.
        """
        self.state.reset_validation(
            split_idx=split_idx, val_loader=dataloader, in_training=False
        )
        self.model.eval()
        dataloader.dataset.eval()
        metrics.reset(reset_df=True)

    def _reset_test(self, dataloader: DataLoader, metrics: MetricsHandler) -> None:
        """
        Resets the relevant objects before a test phase.
        """
        self.state.reset_test(dataloader)
        self.model.eval()
        dataloader.dataset.eval()
        metrics.reset(reset_df=True)

    @staticmethod
    def _seed_context(seed: Optional[int], deterministic: bool) -> ContextManager[None]:
        """
        Returns a context manager for reproducibility if a seed is passed and a
        null context otherwise.
        """
        return (
            seed_everything_context(seed=seed, deterministic=deterministic)
            if seed is not None
            else nullcontext()
        )

    @contextmanager
    def _exception_context(self) -> Generator[None, None, None]:
        """
        Context manager to handle exceptions.
        """
        try:
            yield
        except Exception as e:
            self._call_event(Event.EXCEPTION, exception=e)
            raise

    def _model_to(self, comp_config: ComputationalConfig) -> None:
        """
        Sends the model to the right device and converts to the specified memory format.
        """
        comp_config.check_device()
        self.model.to(device=comp_config.device, non_blocking=comp_config.non_blocking)
        if comp_config.channels_last:
            try:
                self.model.to(memory_format=torch.channels_last)
            except RuntimeError:
                self.model.to(memory_format=torch.channels_last_3d)

    @classmethod
    def _batch_to(cls, batch: BatchType, computational: ComputationalConfig) -> None:
        """
        Sends the data to the right device and converts to the specified memory format.
        """
        if isinstance(batch, Batch):
            batch.to(
                device=computational.device,
                non_blocking=computational.non_blocking,
                channels_last=computational.channels_last,
            )
        elif isinstance(batch, dict):
            for b in batch.values():
                cls._batch_to(b, computational=computational)
        else:  # batch has been checks by ChecksCallback, so it must be a sequence
            for b in batch:
                cls._batch_to(b, computational=computational)

    def _load_model_checkpoint(self, model_path: Path) -> None:
        """
        Loads model weights on CPU.
        """
        self.model.to(CPU)  # load weights on cpu
        state_dict = self.maps.open_file(model_path)
        self.model.load_state_dict(state_dict)

    def _clip_gradients(
        self,
    ) -> None:
        """
        To clip gradients.
        """
        if self.optimization.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), clip_value=self.optimization.clip_grad_value
            )
        if self.optimization.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.optimization.clip_grad_norm,
                norm_type=self.optimization.grad_norm_type,
            )

    def _call_event(self, event: Event, **kwargs):
        """
        Calls a callback event.
        """
        self.callbacks.call_event(
            event, model=self.model, maps=self.maps, state=self.state, **kwargs
        )

    def _get_split(self, split_idx: int) -> Split:
        """
        To load a old split.
        """
        try:
            train_dataset, train_dataloader_config = _get_data(
                self.maps.training.data.train.splits[split_idx]
            )
            val_dataset, val_dataloader_config = _get_data(
                self.maps.training.data.validation.splits[split_idx]
            )
        except CannotReadJsonError as e:
            e.add_note("Please pass directly the split via 'split'.")
            raise

        train_data = self.maps.open_file(
            self.maps.training.data.train.splits[split_idx].data_tsv
        )
        val_data = self.maps.open_file(
            self.maps.training.data.validation.splits[split_idx].data_tsv
        )

        train_dataset = train_dataset.subset(train_data)
        val_dataset = val_dataset.subset(val_data)

        split = Split(
            index=split_idx, train_dataset=train_dataset, val_dataset=val_dataset
        )
        split.build_train_loader(train_dataloader_config)
        split.build_val_loader(val_dataloader_config)

        return split

    def _get_models_in_split(
        self, split_idx: int, model_checkpoint: Optional[str]
    ) -> Iterator[tuple[str, Path]]:
        """
        Gets either the paths to all the models in a split, or the
        path to a specific checkpoint. Returns a generator in both cases.
        """
        if model_checkpoint:
            yield (
                model_checkpoint,
                self.maps.training.splits[split_idx]
                .models.get_checkpoint_dir(model_checkpoint)
                .model_pt,
            )
        else:
            for model_checkpoint in self.maps.training.splits[
                split_idx
            ].models.get_all_models():
                yield from self._get_models_in_split(split_idx, model_checkpoint)

    def _get_dataloader(self, data_dir: DataDir) -> DataLoader:
        """
        To load old dataset and dataloader config and build a dataloader
        with them.
        """
        try:
            dataset, dataloader_config = _get_data(data_dir)
        except CannotReadJsonError as e:
            e.add_note("Please pass directly the dataloader via 'dataloader'.")
            raise

        return dataloader_config.get_object(dataset)

    def _get_old_computational_config(self, split_idx) -> ComputationalConfig:
        """
        Gets the computational setting from an old training.
        """
        return ComputationalConfig.from_json(
            self.maps.training.splits[split_idx].computational_json
        )

    def _create_split(self, split_idx: int) -> None:
        """
        Checks that the split directory doesn't exist and creates it.
        """
        if split_idx in self.maps.training.splits_list:
            raise ValueError(
                f"Training on split {split_idx} has already been performed. To relaunch a training on this split, "
                "first delete it properly with clinicadl.io.Maps.delete_split; to resume a training on this split, "
                "set resume=True."
            )
        self.maps.training.create_split(split_idx, exist_ok=True)

    def _create_results_dir(
        self,
        inference_dir: InferenceDirType,
        group_name: str,
        model_checkpoint: str,
    ) -> None:
        """
        Creates the directory to save the results of the checkpoint on the group.
        """
        split_idx, chkpt = self.maps.training.read_checkpoint_name(model_checkpoint)
        inference_dir.create_group(group_name, exist_ok=True)
        group_dir = inference_dir.groups[group_name].results
        group_dir.create_split(split_idx, exist_ok=True)
        if chkpt in group_dir.splits[split_idx].models_list:
            raise FileExistsError(
                f"There are already some results for checkpoint '{chkpt}' in {group_dir.splits[split_idx].models[chkpt].path}. "
                f"If you want to continue, please first delete the folder."
            )
        group_dir.splits[split_idx].create_model(chkpt, exist_ok=True)

    def _check_split_exists(self, split_idx: int) -> None:
        """
        Checks that a split directory exists.
        """
        if split_idx not in self.maps.training.splits_list:
            raise KeyError(f"No training performed on split {split_idx}.")

    def _check_group_exists(self, group: str) -> None:
        """
        Checks that a group already exists.
        """
        if group not in self.maps.test.groups_list:
            raise KeyError(
                f"The group you passed ('{group}') does not exist yet, so you must pass a dataloader."
            )


def _get_data(data_dir: DataDir) -> tuple[Dataset, DataLoaderConfig]:
    """
    To load an old dataset and dataloader.
    """

    def _error_msg(obj: str, path: Path) -> str:
        return f"ClinicaDL could not read the {obj} in {path}."

    try:
        dataset = get_dataset_from_json(data_dir.dataset_json)
    except Exception as e:
        raise CannotReadJsonError(_error_msg("dataset", data_dir.dataset_json)) from e

    try:
        dataloader_config = DataLoaderConfig.from_json(data_dir.dataloader_json)
    except Exception as e:
        raise CannotReadJsonError(
            _error_msg("dataloader", data_dir.dataloader_json)
        ) from e

    return dataset, dataloader_config
