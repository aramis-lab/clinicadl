.. _api_metrics:

:mod:`clinicadl.metrics.config`
===============================

.. automodule:: clinicadl.metrics.config

.. currentmodule:: clinicadl.metrics.config

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/object_config_class.rst

   LossMetricConfig

Classification
--------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/object_config_class.rst

    ConfusionMatrixMetricConfig
    ROCAUCMetricConfig
    AveragePrecisionMetricConfig

Regression
----------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/object_config_class.rst

    MSEMetricConfig
    MAEMetricConfig
    RMSEMetricConfig

Reconstruction
--------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/object_config_class.rst

    PSNRMetricConfig
    SSIMMetricConfig
    MultiScaleSSIMMetricConfig

Segmentation
------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/object_config_class.rst

    MeanIoUConfig
    DiceMetricConfig
    GeneralizedDiceScoreConfig
    SurfaceDiceMetricConfig
    SurfaceDistanceMetricConfig
    HausdorffDistanceMetricConfig