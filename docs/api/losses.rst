.. _api_losses:

:mod:`clinicadl.losses`
=======================

.. automodule:: clinicadl.losses

:mod:`clinicadl.losses.config`
------------------------------

.. automodule:: clinicadl.losses.config

.. currentmodule:: clinicadl.losses.config


Classification / Segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/config_object.rst

   BCELossConfig
   BCEWithLogitsLossConfig
   CrossEntropyLossConfig
   NLLLossConfig
   MultiMarginLossConfig


Regression / Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/config_object.rst

   MSELossConfig
   L1LossConfig
   SmoothL1LossConfig
   HuberLossConfig
   KLDivLossConfig


MONAI Segmentation
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/config_object.rst

   DiceLossConfig
   DiceCELossConfig
   DiceFocalLossConfig
   GeneralizedDiceLossConfig
   GeneralizedDiceFocalLossConfig
   FocalLossConfig
   TverskyLossConfig
   SoftclDiceLossConfig


MONAI Reconstruction
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: autosummary/config_object.rst

   SSIMLossConfig