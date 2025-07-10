.. _api_data:

:mod:`clinicadl.data`
=====================

.. automodule:: clinicadl.data

:mod:`clinicadl.data.datasets`
------------------------------

.. automodule:: clinicadl.data.datasets

.. currentmodule:: clinicadl.data.datasets

.. autosummary::
   :nosignatures:
   :template: autosummary/class.rst

    CapsDataset
    ConcatDataset
    PairedDataset
    UnpairedDataset

:mod:`clinicadl.data.datatypes`
-------------------------------

.. automodule:: clinicadl.data.datatypes

.. currentmodule:: clinicadl.data.datatypes

.. autosummary::
   :nosignatures:
   :template: autosummary/config_class.rst

    Custom
    T1Linear
    FlairLinear
    PETLinear
    DWIDTI

:mod:`clinicadl.data.dataloader`
--------------------------------

.. automodule:: clinicadl.data.dataloader

.. currentmodule:: clinicadl.data.dataloader

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: autosummary/config_class.rst

    DataLoaderConfig

:mod:`clinicadl.data.structures`
--------------------------------

.. automodule:: clinicadl.data.structures

.. currentmodule:: clinicadl.data.structures

.. autosummary::
    :nosignatures:
    :template: autosummary/class.rst

    DataPoint
    ColinDataPoint

.. toctree::
   :hidden:

   datasets
   datatypes
   structures

:mod:`clinicadl.data.utils`
---------------------------

.. automodule:: clinicadl.data.utils

.. currentmodule:: clinicadl.data.utils

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: autosummary/function.rst

    remove_tensors