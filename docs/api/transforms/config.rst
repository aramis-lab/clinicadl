.. _api_transforms_config:

:mod:`clinicadl.transforms.config`
==================================

.. automodule:: clinicadl.transforms.config

.. currentmodule:: clinicadl.transforms.config

Pre-processing
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/
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
   :toctree: ../generated/
   :nosignatures:
   :template: config_class/class.rst

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

Post-processing
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: config_class/class.rst

   ActivationsConfig
   AsDiscreteConfig
   KeepLargestConnectedComponentConfig
   DistanceTransformEDTConfig
   RemoveSmallObjectsConfig
   LabelFilterConfig
   FillHolesConfig
   SobelGradientsConfig