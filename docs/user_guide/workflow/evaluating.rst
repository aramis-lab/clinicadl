.. _user_guide_workflow_evaluating:

2.3 Evaluating
==============

Training tells you whether your model learns; **evaluation** tells you how well it
performs, and on which data. ClinicaDL evaluates a model by computing **metrics** on
a dataset, either during training (on the validation set) or afterwards (on
validation or held-out test data).

Metrics
-------

A metric is described by a :py:class:`~clinicadl.metrics.Metric`, and the set of
metrics computed during an evaluation phase is gathered in a
:py:class:`~clinicadl.metrics.MetricsHandler`. In practice you rarely manipulate
these directly: you declare the metrics you want when building the
:py:class:`~clinicadl.train.Trainer`, passing them as a dictionary of
:py:mod:`metric configuration objects <clinicadl.metrics.config>`:

.. code-block:: python

    from clinicadl.train import Trainer
    from clinicadl.metrics.config import LossMetricConfig, ConfusionMatrixMetricConfig

    trainer = Trainer(
        maps="maps",
        model=model,
        metrics={
            "loss": LossMetricConfig(loss_name="loss"),
            "f1": ConfusionMatrixMetricConfig(metric_name="f1 score"),
        },
    )

ClinicaDL provides metrics for classification (confusion-matrix metrics, ROC AUC,
average precision), regression (MSE, MAE, RMSE), reconstruction (PSNR, SSIM) and
segmentation (Dice, IoU, Hausdorff distance, …) — see
:py:mod:`clinicadl.metrics.config` for the full list. The metrics defined here are
the ones the ``Trainer`` *can* compute; which ones are actually computed is chosen at
each evaluation call.

Evaluating during and after training
------------------------------------

During training, the metrics are computed on the validation set at the interval set
by the :py:class:`~clinicadl.optim.OptimizationConfig` (see :doc:`Training <training>`).

After training, two methods let you evaluate saved checkpoints. Both identify a
checkpoint by name — ``"best-<metric>"``, ``"epoch-<n>"`` or ``"final"``, prefixed by
the split for :py:meth:`~clinicadl.train.Trainer.test`.

:py:meth:`~clinicadl.train.Trainer.validate`
    Computes **new** metrics on the **validation** data of a split — useful when you
    realise after training that you want an additional metric.

    .. code-block:: python

        trainer.add_metrics(recall=ConfusionMatrixMetricConfig(metric_name="recall"))
        trainer.validate(split_idx=0, metrics=["recall"], model_checkpoint="best-f1")

:py:meth:`~clinicadl.train.Trainer.test`
    Evaluates a checkpoint on a **held-out test** set, identified by a ``group_name``.
    ClinicaDL checks for :term:`data leakage` between the test data and the
    training/validation data.

    .. code-block:: python

        from clinicadl.data.dataloader import DataLoader

        test_dataset = dataset.subset(split_dir / "test_baseline.tsv")
        test_loader = DataLoader(test_dataset)

        trainer.test(
            model_checkpoint="split-0_final",
            group_name="test",
            dataloader=test_loader,
        )

All the results are written to the :term:`MAPS` (see
:doc:`Chapter 3 <../reproducibility/index>`).

Customising inference
---------------------

How an image is fed to the network at evaluation time is governed by an
:py:class:`~clinicadl.infer.Inferer`, attached to the model through its ``inferer``
argument (see :doc:`Defining a model <model>`). ClinicaDL provides:

- :py:class:`~clinicadl.infer.SimpleInferer` — the default; the whole image is passed
  through the network, with optional post-processing of the output;
- :py:class:`~clinicadl.infer.PatchesToImageInferer` — splits a volume into 3D
  patches, runs the network on each, and merges the outputs back into a full volume;
- :py:class:`~clinicadl.infer.SlicesToImageInferer` — does the same with 2D slices and
  a 2D network.

.. code-block:: python

    from clinicadl.models import SupervisedModel
    from clinicadl.infer import PatchesToImageInferer

    model = SupervisedModel(
        network=...,
        loss=...,
        optimizer=...,
        inferer=PatchesToImageInferer(patch_size=64, overlap=0.25),
        label_key="mask",
    )

Inferers also handle **post-processing** (e.g. activations, thresholding) applied to
the network's output before metrics are computed.

----

Evaluation done, the :doc:`last section <callbacks>` of this chapter shows how to
customise the training loop with callbacks.
