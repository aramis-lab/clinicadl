.. _user_guide_workflow_callbacks:

2.4. Callbacks
==============

A :py:class:`~clinicadl.train.Trainer` always runs the same steps, but at many
points of that sequence you may want to *do something extra* — log a message, save a
checkpoint, stop early, schedule the learning rate. These non-essential actions are
the job of **callbacks**.

.. important::

    Callbacks capture **non-essential** logic (logging, checkpointing, etc.). The
    *essential* logic of your experiment — how a batch is processed, how the loss is
    computed — belongs in your :py:class:`~clinicadl.models.Model` (see
    :doc:`Defining a model <model>`).

A :py:class:`~clinicadl.callbacks.Callback` defines actions to perform at specific
events of the training and evaluation workflows (the start of an epoch, the end of a
batch, the moment metrics are computed, etc.). You attach callbacks to a ``Trainer``
through its ``callbacks`` argument.

Here is an example of a ``Trainer`` that stops training after 10 evaluation steps without
improvement in the validation loss, saves the best model with respect
to the validation loss, and stores model checkpoints every 10 epochs.

.. code-block:: python

    from clinicadl.train import Trainer
    from clinicadl.callbacks import EarlyStoppingCallback, ModelCheckpointCallback

    trainer = Trainer(
        maps="maps",
        model=model,
        metrics={"loss": LossMetricConfig(loss_name="loss")},
        callbacks=[
            EarlyStoppingCallback(metric="loss", patience=10),
            ModelCheckpointCallback(metric="loss", epochs=range(1, 100, 10)),
        ],
    )

Available callbacks
-------------------

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Callback
      - Role
    * - :py:class:`~clinicadl.callbacks.EarlyStoppingCallback`
      - Stops training when a monitored metric stops improving.
    * - :py:class:`~clinicadl.callbacks.LRSchedulerCallback`
      - Adjusts the learning rate during training.
    * - :py:class:`~clinicadl.callbacks.ModelCheckpointCallback`
      - Saves the network weights at chosen epochs and/or the best model w.r.t. a metric.
    * - :py:class:`~clinicadl.callbacks.TrainingCheckpointCallback`
      - Saves the training state so that an interrupted training can be
        :ref:`resumed <user_guide_workflow_resuming>`.
    * - :py:class:`~clinicadl.callbacks.LoggerCallback`
      - Configures logging (console and files) and the progress bar.
    * - :py:class:`~clinicadl.callbacks.MonitorCallback`
      - Records computational statistics (time, GPU memory, throughput).

.. note::

    Some callbacks are active **by default** —
    :py:class:`~clinicadl.callbacks.LoggerCallback`,
    :py:class:`~clinicadl.callbacks.MonitorCallback`,
    :py:class:`~clinicadl.callbacks.ModelCheckpointCallback` and
    :py:class:`~clinicadl.callbacks.TrainingCheckpointCallback`. To change their
    settings, pass a new instance with the parameters you want; it overrides the
    default one. All the callbacks are managed by a
    :py:class:`~clinicadl.callbacks.CallbacksHandler`, in which **order matters**.

Writing your own callback
-------------------------

When the built-in callbacks are not enough, you can write your own by subclassing
:py:class:`~clinicadl.callbacks.Callback` and overriding the ``on_...`` methods
associated with the events you care about:

.. code-block:: python

    from clinicadl.callbacks import Callback

    class PrintEpochCallback(Callback):
        def on_epoch_start(self, *, model, maps, state) -> None:
            print(f"Starting epoch {state.current_epoch}")

Writing your own ClinicaDL objects — callbacks and others — is the subject of
:doc:`Chapter 4 <../customising/index>`.

----

This closes Chapter 2: you can now define a model, train it, evaluate it, and
customise you workflow. The :doc:`next chapter <../reproducibility/index>` explains how
ClinicaDL helps you manage and reproduce your experiments.
