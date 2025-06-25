Preprocessing and Data Augmentation
===================================

Supported Transforms
--------------------

Many `TorchIO <https://torchio.readthedocs.io/index.html>`_ transforms are natively supported in ClinicaDL and
can thus be passed to :py:class:`Transforms <clinicadl.transforms.Transforms>` via configuration classes.
Checkout out the documentation on :ref:`preprocessing transforms <api_preprocessing>` and
:ref:`augmentations <api_augmentation>` to know the catalogue of natively supported transforms
in ClinicaDL.

Nevertheless, you can also pass your own transform to :py:class:`Transforms <clinicadl.transforms.Transforms>`
(see the :tutorials:`tutorial <transforms/custom_transforms.ipynb>`), or use the transforms implemented by
the community in :zoo:`ClinicaDL Zoo <>`.