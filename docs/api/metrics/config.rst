.. _api_metrics:

:mod:`clinicadl.metrics.config`
===============================

.. automodule:: clinicadl.metrics.config

.. currentmodule:: clinicadl.metrics.config

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/pydantic.rst

   LossMetricConfig

Classification
--------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/pydantic.rst

    ConfusionMatrixMetricConfig
    ROCAUCMetricConfig
    AveragePrecisionMetricConfig

Regression
----------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/pydantic.rst

    MSEMetricConfig
    MAEMetricConfig
    RMSEMetricConfig

Reconstruction
--------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/pydantic.rst

    PSNRMetricConfig
    SSIMMetricConfig
    MultiScaleSSIMMetricConfig

Segmentation
------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/pydantic.rst

    MeanIoUConfig
    DiceMetricConfig
    GeneralizedDiceScoreConfig
    SurfaceDiceMetricConfig
    SurfaceDistanceMetricConfig
    HausdorffDistanceMetricConfig