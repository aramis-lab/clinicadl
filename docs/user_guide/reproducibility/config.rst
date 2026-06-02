.. _user_guide_reproducibility_config:

3.1 Configuration classes
=========================

Throughout this guide we have passed raw objects to ClinicaDL — TorchIO transforms,
PyTorch losses and networks. This is convenient, but raw objects have a drawback for
reproducibility: ClinicaDL cannot always inspect them to know exactly how they were
parametrised, nor rebuild them later from a saved experiment.

This is the problem **configuration classes** solve.

What is a configuration class?
------------------------------

A configuration class is a small dataclass associated with a ClinicaDL object. It
holds **only the parameters** of that object, with almost no logic of its own. From a
configuration object, you obtain the actual object with its ``get_object`` method.

For instance, instead of building an optimizer directly, you describe it with
:py:class:`~clinicadl.optim.optimizers.config.AdamConfig`:

.. code-block:: python

    from clinicadl.optim.optimizers.config import AdamConfig

    config = AdamConfig(lr=1e-4)

    optimizer = config.get_object(network=network)   # the underlying torch.optim.Adam

Because a configuration class contains only data, it can be **saved and read back**,
which a raw object generally cannot:

.. code-block:: python

    config.to_json("adam.json")
    config = AdamConfig.from_json("adam.json")

This is exactly what makes an experiment reproducible: ClinicaDL stores the
configuration of every object it uses, so the whole setup can be rebuilt later (see
:doc:`MAPS <maps>`).

Where you can use them
----------------------

Wherever ClinicaDL accepts a raw object, it usually also accepts the matching
configuration class. Configuration classes exist for the main building blocks of the
library:

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Module
      - Covers
    * - :py:mod:`clinicadl.networks.config`
      - Neural networks (e.g. :py:class:`~clinicadl.networks.config.ResNet18Config`)
    * - :py:mod:`clinicadl.losses.config`
      - Loss functions (e.g. :py:class:`~clinicadl.losses.config.CrossEntropyLossConfig`)
    * - :py:mod:`clinicadl.optim.optimizers.config`
      - Optimizers (e.g. :py:class:`~clinicadl.optim.optimizers.config.AdamConfig`)
    * - :py:mod:`clinicadl.optim.lr_schedulers.config`
      - Learning-rate schedulers (e.g. :py:class:`~clinicadl.optim.lr_schedulers.config.OneCycleLRConfig`)
    * - :py:mod:`clinicadl.transforms.config`
      - Transforms — preprocessing, augmentation, post-processing
    * - :py:mod:`clinicadl.metrics.config`
      - Metrics (e.g. :py:class:`~clinicadl.metrics.config.DiceMetricConfig`)

As an example, a :py:class:`~clinicadl.transforms.TransformsHandler` can be built from
raw TorchIO transforms or from their configuration classes — the two are equivalent,
but only the second is fully reproducible:

.. code-block:: python

    import torchio as tio
    from clinicadl.transforms import TransformsHandler
    from clinicadl.transforms.config import ZNormalizationConfig

    # with a raw transform ...
    transforms = TransformsHandler(image_transforms=[tio.ZNormalization()])

    # ... or with its configuration class
    transforms = TransformsHandler(image_transforms=[ZNormalizationConfig()])

.. tip::

    To get the **most reproducible** experiments, prefer configuration classes over
    raw objects wherever one exists.

.. note::

    The list of available configuration classes is **not exhaustive** — it covers
    mostly the objects coming from :monai:`MONAI <>` and :torchio:`TorchIO <>` — but it
    keeps growing. When no configuration class exists for an object, you can still use
    the raw object; only that object will not be reproducible by ClinicaDL.

----

Configuration classes are what allows the :doc:`MAPS <maps>` to record a complete,
reproducible description of your experiment.
