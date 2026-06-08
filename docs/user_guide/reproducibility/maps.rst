.. _user_guide_reproducibility_maps:

3.2. MAPS
=========

A **MAPS** (Model Analysis and Processing Structure) is the cornerstone of experiment
management in ClinicaDL: a single directory that holds *everything* about an
experiment. It is represented in Python by the :py:class:`~clinicadl.io.maps.Maps`
class.

What a MAPS contains
--------------------

Whenever you train, validate or test a model, the
:py:class:`~clinicadl.train.Trainer` writes to its MAPS:

- **Outputs of training and evaluation** — the trained network weights and
  checkpoints, the computed metrics, the logs, computational statistics, and the
  results of testing on new data.
- **All the hyperparameters used** — the model, the optimization settings, the computational settings, the
  callbacks, the metrics and the data splits.

Because everything lives in one place, a MAPS serves the two goals of this chapter at
once:

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: :fas:`box-archive` Experiment management

        One self-contained, **shareable** folder fully describes one experiment —
        results, logs and configuration together. No more scattered output files.

    .. grid-item-card:: :fas:`rotate` Reproducibility

        Since the MAPS records the configuration of every object used, an experiment
        can be **rebuilt and run again** from the folder alone.

Creating and reopening a MAPS
-----------------------------

You rarely create a MAPS by hand: a :py:class:`~clinicadl.train.Trainer` creates one
for you from the path you give it (see :doc:`Chapter 2 <../workflow/index>`).

.. code-block:: python

    from clinicadl.train import Trainer
    from clinicadl.metrics.config import LossMetricConfig, MSEMetricConfig
    from clinicadl.callbacks import ModelCheckpointCallback

    trainer = Trainer(
        maps="maps",
        model=model,
        metrics={"loss": LossMetricConfig(), "mse": MSEMetricConfig()},
        callbacks=[ModelCheckpointCallback(metric="loss", epochs=range(1, 100, 10))],
    )
    trainer.train(split)
    trainer.test(
        model_checkpoint="split-0_best-loss", group_name="test", dataloader=test_loader
    )
    trainer.test(
        model_checkpoint="split-0_epoch-10", group_name="test"
    )

will produce a MAPS like this:

.. code-block:: text

    maps
    ├── environment.txt
    ├── metrics.json
    ├── model.json
    ...
    ├── test
    │   └── group-test
    │       ├── data.tsv
    │       ├── dataloader.json
    │       ├── dataset.json
    │       └── results
    │           └── split-0
    │               ├── best-loss
    │               │   ├── computational.json
    │               │   └── metrics
    │               │       ├── aggregated.tsv
    │               │       └── details.tsv
    │               └── epoch-10
    └── training
        ├── data
        │   ├── data.tsv
        │   ├── train
        │   │   ├── split-0
        │   │   │   ├── data.tsv
        │   │   │   ├── dataloader.json
        │   │   │   └── dataset.json
        │   │   └── split-1
        │   └── validation
        │       ├── split-0
        │       └── split-1
        ├── optimization.json
        └── split-0
            ├── computational.json
            ├── logs
            │   ├── computational.tsv
            │   ├── training_loss.tsv
            │   ...
            ├── models
            │   ├── best_models
            │   │   └── best-loss
            │   │       ├── model.pt
            │   │       └── validation_metrics
            │   ├── checkpoints
            │   │   ├── epoch-1
            │   │   ├── epoch-10
            │   │   ...
            │   └── final
            ├── summary.log
            ├── validation_metrics
            └── warning.log

Later you can reopen the experiment.
:py:meth:`Trainer.from_maps <clinicadl.train.Trainer.from_maps>` rebuilds the whole
trainer from the configurations stored in the MAPS:

.. code-block:: python

    trainer = Trainer.from_maps("maps")

To inspect the folder directly, open it as a :py:class:`~clinicadl.io.maps.Maps`,
which exposes the path of every file and subdirectory it contains:

.. code-block:: python

    from clinicadl.io.maps import Maps

    maps = Maps("maps")

.. code-block:: python

    >>> best_metrics_path = maps.training.splits[0].models.best_models.metrics["loss"].validation_metrics.aggregated_tsv
    >>> best_metrics_path
    PosixPath('maps/training/split-0/models/best_models/best-loss/validation_metrics/aggregated.tsv')
    >>> maps.open_file(best_metrics_path)
        epoch       mse      loss
    0      42  1.347617  0.853633

.. warning::

    Reproducibility relies on :doc:`configuration classes <config>`. If you used a
    **raw object** that has no configuration class (a custom transform, an external
    network, etc.), ClinicaDL may not be able to rebuild it. Prefer configuration classes whenever one exists.

----

This closes Chapter 3. The :doc:`final chapter <../customising/index>` shows how to
go beyond the ready-made objects and tailor ClinicaDL to your own needs.
