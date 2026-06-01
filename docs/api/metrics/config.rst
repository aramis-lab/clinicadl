.. _api_metrics_config:

:mod:`clinicadl.metrics.config`
===============================

.. automodule:: clinicadl.metrics.config

.. currentmodule:: clinicadl.metrics.config

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/config_object.rst

   LossMetricConfig

Classification
--------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/config_object.rst

    ConfusionMatrixMetricConfig
    ROCAUCMetricConfig
    AveragePrecisionMetricConfig

Regression
----------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/config_object.rst

    MSEMetricConfig
    MAEMetricConfig
    RMSEMetricConfig

Reconstruction
--------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/config_object.rst

    PSNRMetricConfig
    SSIMMetricConfig
    MultiScaleSSIMMetricConfig

Segmentation
------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/config_object.rst

    MeanIoUConfig
    DiceMetricConfig
    GeneralizedDiceScoreConfig
    SurfaceDiceMetricConfig
    SurfaceDistanceMetricConfig
    HausdorffDistanceMetricConfig