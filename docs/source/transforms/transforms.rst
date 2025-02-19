.. _transforms:

Transforms
==========
.. autoclass:: clinicadl.transforms.Transforms

Extraction
----------
.. currentmodule:: clinicadl.transforms.extraction

Image
*****
.. autoclass:: Image

Patch
*****
.. autoclass:: Patch

Slice
*****
.. autoclass:: Slice

Supported Transforms
--------------------

.. autofunction:: clinicadl.transforms.get_transform_config

Use ``get_transform_config`` to get any :ref:`transforms_preprocessing` or  :ref:`transforms_augmentation` transform.

.. toctree::
    :maxdepth: 1

    preprocessing
    augmentation