.. _user_guide_customising:

4. Customising your ClinicaDL experiment
=========================================

ClinicaDL ships many ready-to-use objects, but no library can anticipate every need.
Its second guiding principle — **flexibility** — means that almost every object can be
*extended*. This short chapter points to the main extension points so that you can
tailor ClinicaDL to your own experiments.

There are two complementary ways to customise ClinicaDL: plugging in **callbacks**,
and **subclassing** its objects.

Callbacks
---------

The lightest way to alter the behaviour of a training run is to add a
:py:class:`~clinicadl.callbacks.Callback`. Callbacks let you run arbitrary,
non-essential actions at specific moments of the training and evaluation workflows —
logging, checkpointing, early stopping, or anything you implement yourself — without
touching the rest of the pipeline. Callbacks are covered in
:doc:`Section 2.4 <../workflow/callbacks>`.

Subclassing ClinicaDL objects
-----------------------------

ClinicaDL is built with object-oriented programming in mind: most of its base classes
are designed to be **inherited from**. By subclassing one of them and overriding a
method, you change a specific behaviour while keeping everything else — and the object
keeps working with the rest of the library.

The most common extension points are:

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Base class
      - Subclass it to …
    * - :py:class:`~clinicadl.models.Model`
      - define your own training/evaluation logic (custom forward step, several
        networks, custom optimization, …).
    * - :py:class:`~clinicadl.data.datasets.Dataset`
      - read data that does not fit the built-in datasets (see
        :ref:`Section 1.2 <user_guide_data_bids>`).
    * - :py:class:`~clinicadl.metrics.Metric`
      - implement a metric that ClinicaDL does not provide.
    * - :py:class:`~clinicadl.infer.Inferer`
      - define a custom inference strategy.
    * - :py:class:`~clinicadl.data.dataloader.CollateFn`
      - control how samples are assembled into a :py:class:`~clinicadl.data.dataloader.Batch`.
    * - :py:class:`~clinicadl.callbacks.Callback`
      - add custom actions during training and evaluation.

Customising a model is the most frequent case. Rather than implementing a
:py:class:`~clinicadl.models.Model` from scratch — which requires defining the whole
training and evaluation logic — it is usually easier to inherit from an existing model
and override only the method you need:

.. code-block:: python

    import torch
    from clinicadl.models import SupervisedModel

    class MyModel(SupervisedModel):
        def forward_step(self, batch) -> torch.Tensor:
            # custom logic to compute the loss from a batch
            images = batch.get_field("image", dtype=torch.float32)
            labels = batch.get_field(self.label_key, ensure_channel_dim=True, dtype=torch.float32)
            outputs = self.network(images)
            return self.loss(outputs, labels)

The simplest extension point of all needs no subclassing: a **transform** is just a
callable that takes and returns a :py:class:`~clinicadl.data.structures.DataPoint`, so
any function of yours can join a :py:class:`~clinicadl.transforms.TransformsHandler`
(see :ref:`Section 1.3.2 <user_guide_data_transforms_pipeline>`).

.. tip::

    Whenever you write a custom object, remember the trade-off from
    :doc:`Chapter 3 <../reproducibility/index>`: objects without a
    :doc:`configuration class <../reproducibility/config>` are not automatically
    reproducible. If reproducibility matters for your custom object, consider giving it
    a configuration class as well.

----

This is the end of the User Guide. You now have an overview of the whole library, from
manipulating data to building, training and managing a reproducible deep learning
experiment. For the precise signature of any object, head to the
:doc:`API Reference <../../api/index>`.
