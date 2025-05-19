.. _splitter:

Splits
======

To split a :py:class:`~clinicadl.data.datasets.CapsDataset` (or its derivatives:
:py:class:`~clinicadl.data.datasets.ConcatDataset`, :py:class:`~clinicadl.data.datasets.PairedDataset`, and
:py:class:`~clinicadl.data.datasets.UnpairedDataset`), you will first find a split from a
:py:class:`~pandas.DataFrame` containing a list of (participant, session) pairs,
and then use this split to split the dataset.

1. To find a split from a DataFrame, you may be interested in:

- :py:func:`~clinicadl.splitter.make_split`: to find a single training-validation/test split; 
- :py:func:`~clinicadl.splitter.make_kfold`: to find multiple training-validation splits for K-Fold
  cross-validation.

.. code:: python

    ici du code

2. Then, to split a dataset between a training and a validation sets, you will need to read the splits you made with ``make_split``
or ``make_kfold``, using :py:class:`~clinicadl.splitter.SingleSplit` or
:py:class:`~clinicadl.splitter.KFold` respectively.

Both ``SingleSplit`` and ``KFold`` will return a :py:class:`~clinicadl.splitter.Split` object,
containing the training and validation sets. :py:class:`~clinicadl.splitter.Split` is the only
data format accepted by ClinicaDL's ``Trainer``.

.. code:: python

    ici du code

.. note::
    If you want to split your dataset between a training and a test set, without having to
    manipulate a :py:class:`~clinicadl.splitter.Split` object, you can just create two different
    datasets:

    .. code::

        ici du code

    Or use :py:meth:`CapsDataset.subset() <clinicadl.data.datasets.CapsDataset.subset>`:

    .. code ::

        ici du code

.. toctree::
    :maxdepth: 1

    single_split
    k_fold