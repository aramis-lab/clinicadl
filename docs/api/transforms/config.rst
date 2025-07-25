.. _api_transforms_config:

:mod:`clinicadl.transforms.config`
==================================

.. automodule:: clinicadl.transforms.config
   :no-index:

.. currentmodule:: clinicadl.transforms.config

.. _api_preprocessing:

Pre-processing
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: config_class/class.rst

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


.. _api_augmentation:

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


.. _api_post_processing:

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