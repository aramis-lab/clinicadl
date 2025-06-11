Optimizers
==========
.. currentmodule:: clinicadl.metrics.config

Natively supported metrics in ClinicaDL
(all from `monai.metrics <https://docs.monai.io/en/stable/metrics.html>`_):

(TODO: loss as a metric)

Classification
--------------

ConfusionMatrix
***************
.. autoclass:: ConfusionMatrixMetricConfig

ROCAUCMetric
************
.. autoclass:: ROCAUCMetricConfig

Regression
----------

MSEMetric
*********
.. autoclass:: MSEMetricConfig

MAEMetric
*********
.. autoclass:: MAEMetricConfig

RMSEMetric
**********
.. autoclass:: RMSEMetricConfig

Segmentation
------------

MeanIoU
*******
.. autoclass:: MeanIoUConfig

DiceMetric
**********
.. autoclass:: DiceMetricConfig

GeneralizedDiceScore
********************
.. autoclass:: GeneralizedDiceScoreConfig

SurfaceDiceMetric
*****************
.. autoclass:: SurfaceDiceMetricConfig

SurfaceDistanceMetric
*********************
.. autoclass:: SurfaceDistanceMetricConfig

HausdorffDistanceMetric
***********************
.. autoclass:: HausdorffDistanceMetricConfig

Reconstruction
--------------

PSNRMetric
**********
.. autoclass:: PSNRMetricConfig

SSIMMetric
**********
.. autoclass:: SSIMMetricConfig

MultiScaleSSIMMetric
********************
.. autoclass:: MultiScaleSSIMMetricConfig