.. _user_guide_customising:

4. Customising your ClinicaDL experiment
=========================================

ClinicaDL comes with a range of ready-to-use components, but no library can cover every possible use case.
One of its core principles — **flexibility** — ensures that most built-in objects can be
*extended* to meet specific or unforeseen needs. This short chapter highlights the main extension points
so you can adapt ClinicaDL to your own experiments.

Using external objects
----------------------

Before extending ClinicaDL, it is worth remembering that at
many points ClinicaDL accepts objects coming from the wider ecosystem —
:torch:`PyTorch <>`, :monai:`MONAI <>`, :torchio:`TorchIO <>` — or objects you wrote
yourself, as long as they follow the expected interface.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Object
      - What you can pass
    * - Neural networks
      - any :py:class:`torch.nn.Module` — from :torchvision:`torchvision <models.html>`, :monai:`MONAI <networks.html>`, or
        your own — given to a :py:class:`~clinicadl.models.Model` (if you work with 3D images, make sure that your neural
        network is 3D!).
    * - Losses
      - any PyTorch-style loss — from :torch:`PyTorch <nn.html#loss-functions>`,
        :monai:`MONAI <losses.html#loss-functions>`, or your own — given to a
        :py:class:`~clinicadl.models.Model`.
    * - Optimizers
      - any :py:class:`torch.optim.Optimizer`, given to a
        :py:class:`~clinicadl.models.Model`.
    * - Learning-rate schedulers
      - any :py:class:`torch.optim.lr_scheduler.LRScheduler`, given to an
        :py:class:`~clinicadl.callbacks.LRSchedulerCallback`.
    * - Transforms
      - any callable taking and returning a
        :py:class:`~clinicadl.data.structures.DataPoint` (e.g. a
        :torchio:`TorchIO transform <transforms>`), given to a
        :py:class:`~clinicadl.transforms.TransformsHandler`.

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
        networks, custom optimization, etc.).
    * - :py:class:`~clinicadl.data.datasets.Dataset`
      - read data that does not fit the built-in datasets.
    * - :py:class:`~clinicadl.data.dataloader.CollateFn`
      - control how samples are assembled into batches.
    * - :py:class:`~clinicadl.metrics.Metric`
      - implement a metric that ClinicaDL does not provide.
    * - :py:class:`~clinicadl.infer.Inferer`
      - define a custom inference strategy.
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

.. tip::

    Whenever you write a custom object, remember the trade-off from
    :doc:`Chapter 3 <../reproducibility/index>`: objects without a
    :doc:`configuration class <../reproducibility/config>` are not automatically
    reproducible.

    If you believe your object could benefit community, consider
    :doc:`contributing <../../contributing>` to have it added to the list of
    objects natively supported by ClinicaDL.

----

This concludes the User Guide. You now have an overview of the whole library, from
manipulating data to building, training and managing a reproducible deep learning
experiment. For the precise signature of any object, head to the
:doc:`API Reference <../../api/index>`.
