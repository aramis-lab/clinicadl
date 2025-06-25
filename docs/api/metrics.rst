.. _api_metrics:

:mod:`clinicadl.metrics`
========================

.. automodule:: clinicadl.metrics

:mod:`clinicadl.metrics.config`
-------------------------------

.. automodule:: clinicadl.metrics.config

.. currentmodule:: clinicadl.metrics.config

Classification
--------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/class.rst

    ConfusionMatrixMetricConfig
    ROCAUCMetricConfig

Regression
----------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/class.rst

    MSEMetricConfig
    MAEMetricConfig
    RMSEMetricConfig

Reconstruction
--------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: class.rst

    PSNRMetricConfig
    SSIMMetricConfig
    MultiScaleSSIMMetricConfig

Segmentation
------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: class.rst

    MeanIoUConfig
    DiceMetricConfig
    GeneralizedDiceScoreConfig
    SurfaceDiceMetricConfig
    SurfaceDistanceMetricConfig
    HausdorffDistanceMetricConfig