.. _transforms:

Transforms
==========
.. autoclass:: clinicadl.transforms.Transforms

.. _extraction:

Extraction
----------
.. currentmodule:: clinicadl.transforms.extraction

Patch
*****
.. autoclass:: Patch

Slice
*****
.. autoclass:: Slice

.. _supported_transforms:

Supported Transforms
--------------------

Many `TorchIO <https://torchio.readthedocs.io/index.html>`_ transforms are natively supported in ClinicaDL and
can thus be passed to :py:class:`Transforms <clinicadl.transforms.Transforms>` via configuration classes.
Checkout out the documentation on :ref:`preprocessing transforms <transforms_preprocessing>` and
:ref:`augmentations <transforms_augmentation>` to know the catalogue of natively supported transforms
in ClinicaDL.

.. toctree::
    :maxdepth: 1

    preprocessing
    augmentation