.. _user_guide_workflow_training:

2.2. Training
=============

Training is orchestrated by the :py:class:`~clinicadl.train.Trainer`. It ties
together a :py:class:`~clinicadl.models.Model`, the data of a
:py:class:`~clinicadl.split.Split`, the metrics to monitor and the callbacks to run,
and takes care of the training loop — moving data to the GPU, mixed precision,
gradient accumulation, evaluation, checkpointing, etc. — so that you only provide the
pieces specific to your experiment.

A first training
----------------

Putting together what we built in :doc:`Chapter 1 <../data/index>` and in
:doc:`Defining a model <model>`:

.. code-block:: python

    from clinicadl.train import Trainer

    # `model` is a Model and `split` a Split (see the previous sections)
    trainer = Trainer(maps="maps", model=model)
    trainer.train(split)

The ``Trainer`` writes everything it produces — trained weights, metrics, logs and
the configuration used — into the ``maps`` directory (the :term:`MAPS`, see
:doc:`Chapter 3 <../reproducibility/index>`).

:py:meth:`~clinicadl.train.Trainer.train` runs the training on a single split. To
train on every fold of a :py:class:`~clinicadl.split.KFold`, simply loop over the
splits:

.. code-block:: python

    for split in splitter.get_splits(dataset):
        split.build_train_loader(batch_size=8, shuffle=True)
        split.build_val_loader(batch_size=8)
        trainer.train(split)

Configuring the optimization
----------------------------

How the optimization is run — the number of epochs, gradient accumulation, gradient
clipping, how often to evaluate — is described by an
:py:class:`~clinicadl.optim.OptimizationConfig`, passed to the ``Trainer``:

.. code-block:: python

    from clinicadl.optim import OptimizationConfig
    from clinicadl.train import Trainer

    trainer = Trainer(
        maps="maps",
        model=model,
        optimization=OptimizationConfig(
            num_epochs=100,
            accumulation_steps=2,      # virtually doubles the batch size
            evaluation_interval=5,     # evaluate every 5 epochs
        ),
    )

Controlling the hardware
------------------------

Computational aspects — GPU, :term:`AMP`, memory format, and the seed for
reproducibility — are set per training run through a
:py:class:`~clinicadl.train.ComputationalConfig` passed to
:py:meth:`~clinicadl.train.Trainer.train`:

.. code-block:: python

    from clinicadl.train import ComputationalConfig

    trainer.train(
        split,
        computational=ComputationalConfig(gpu=True, amp=True, seed=42, deterministic=True),
    )

.. tip::

    Setting ``seed`` and ``deterministic=True`` makes a training run reproducible.
    A global seed can also be set once with
    :py:func:`clinicadl.utils.seed.seed_everything` or :py:func:`clinicadl.utils.seed.seed_everything_context`.

Monitoring the training
-----------------------

Monitoring the training is done through **metrics** and **callbacks** (logging, early stopping,
etc.). We cover metrics in
:doc:`Evaluating <evaluating>` and callbacks in :doc:`Callbacks <callbacks>`.

.. _user_guide_workflow_resuming:

2.2.1. Resuming an interrupted training
---------------------------------------

Long trainings can be interrupted — a bug, a power cut. As long
as a :py:class:`~clinicadl.callbacks.TrainingCheckpointCallback` was active (it is one
of the default callbacks), the ``Trainer`` periodically saves a checkpoint of the
training state in the :term:`MAPS`, and you can pick up where it stopped with
:py:meth:`~clinicadl.train.Trainer.resume`:

.. code-block:: python

    trainer.resume(split_idx=0)

If the ``Trainer`` object is no longer in memory — typically in a fresh Python
session — rebuild it from the :term:`MAPS` first, then resume:

.. code-block:: python

    from clinicadl.train import Trainer

    trainer = Trainer.from_maps("maps")
    trainer.resume(split_idx=0)

.. important::

    The computational setup recorded at training time is reused on resume. If the
    model was first trained on a GPU, make sure a GPU is available when you resume.

----

Now you know how to setup the training of a model. The :doc:`next section <evaluating>` shows how to evaluate it.
