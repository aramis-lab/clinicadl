.. _api_losses:

``clinicadl.losses``
====================
.. currentmodule:: clinicadl.losses.config

Natively supported loss functions in ClinicaDL
(all from `torch.nn <https://pytorch.org/docs/stable/nn.html#loss-functions>`_):


Classification / Segmentation
-----------------------------

BCELoss
*******
.. autoclass:: BCELossConfig

BCEWithLogitsLoss
*****************
.. autoclass:: BCEWithLogitsLossConfig

CrossEntropyLoss
****************
.. autoclass:: CrossEntropyLossConfig

NLLLoss
*******
.. autoclass:: NLLLossConfig

MultiMarginLoss
***************
.. autoclass:: MultiMarginLossConfig

Regression / Reconstruction
---------------------------

MSELoss
*******
.. autoclass:: MSELossConfig

L1Loss
******
.. autoclass:: L1LossConfig

SmoothL1Loss
************
.. autoclass:: SmoothL1LossConfig

HuberLoss
*********
.. autoclass:: HuberLossConfig

KLDivLoss
*********
.. autoclass:: KLDivLossConfig