LR schedulers
==============
.. currentmodule:: clinicadl.optim.lr_schedulers.config

Natively supported learning rate schedulers in ClinicaDL
(all from `torch.optim <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_):

(TODO: parameter groups)

ConstantLR
**********
.. autoclass:: ConstantLRConfig

ExponentialLR
*************
.. autoclass:: ExponentialLRConfig

LinearLR
********
.. autoclass:: LinearLRConfig

PolynomialLR
************
.. autoclass:: PolynomialLRConfig

StepLRConfig
************
.. autoclass:: StepLRConfig

MultiStepLRConfig
*****************
.. autoclass:: MultiStepLRConfig

ReduceLROnPlateau
*****************
.. autoclass:: ReduceLROnPlateauConfig

OneCycleLR
**********
.. autoclass:: OneCycleLRConfig