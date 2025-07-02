.. _api_transforms:

:mod:`clinicadl.transforms`
===========================

.. automodule:: clinicadl.transforms

.. currentmodule:: clinicadl.transforms

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/config_class.rst

   Transforms

:mod:`clinicadl.transforms.extraction`
--------------------------------------

.. automodule:: clinicadl.transforms.extraction

.. currentmodule:: clinicadl.transforms.extraction

.. autosummary::
   :nosignatures:
   :template: autosummary/config_class.rst

   Image
   Patch
   Slice

:mod:`clinicadl.transforms.config`
----------------------------------

.. automodule:: clinicadl.transforms.config

.. currentmodule:: clinicadl.transforms.config


Preprocessing
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :template: autosummary/config_class.rst

   RescaleIntensityConfig
   ZNormalizationConfig
   MaskConfig
   ClampConfig
   CropOrPadConfig
   ToCanonicalConfig
   ResizeConfig
   ResampleConfig
   EnsureShapeMultipleConfig
   CropConfig
   PadConfig
   RemapLabelsConfig
   OneHotConfig

Augmentation
^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :template: autosummary/config_class.rst

   OneOfConfig
   RandomMotionConfig
   RandomGhostingConfig
   RandomSpikeConfig
   RandomBiasFieldConfig
   RandomBlurConfig
   RandomNoiseConfig
   RandomSwapConfig
   RandomGammaConfig
   RandomFlipConfig
   RandomAffineConfig
   RandomElasticDeformationConfig
   RandomAnisotropyConfig

.. toctree::
   :hidden:

   extraction
   config