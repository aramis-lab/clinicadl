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

    from clinicadl.splitter import make_split, make_kfold

    # 'mycaps/participants_sessions.tsv' looks like this
    #     participant_id	session_id	age sex	diagnosis
    # 0	         sub-000	  ses-M000	 60	  F        CN
    # 1	         sub-000	  ses-M003	 60	  F        AD
    # 2	         sub-010	  ses-M003	 55	  M        AD
    # ...
    
    # let's build our training and test sets
    split_dir = make_split(
        data="mycaps/participants_sessions.tsv",
        output_dir="mycaps/splits",
        stratification=["diagnosis", "age", "sex"],
        n_test=0.2,
    )

    # now let's perform a 5-fold split on our training set
    kfold_dir = make_kfold(
        data=split_dir / "train.tsv",
        n_splits=5,
        longitudinal=True,
    )

2. Then, to split a dataset between a training and a validation sets, you will need to read the splits you made with ``make_split``
or ``make_kfold``, using :py:class:`~clinicadl.splitter.SingleSplit` or
:py:class:`~clinicadl.splitter.KFold` respectively.

Both ``SingleSplit`` and ``KFold`` will return a :py:class:`~clinicadl.splitter.Split` object,
containing the training and validation sets. :py:class:`~clinicadl.splitter.Split` is the only
data format accepted by ClinicaDL's ``Trainer``.

.. code:: python

    from clinicadl.splitter import KFold, SingleSplit
    from clinicadl.data.datasets import CapsDataset

    dataset = CapsDataset("mycaps", data="mycaps/participants_sessions.tsv")

    splitter = SingleSplit(split_dir)     # read the split
    split = splitter.get_split(dataset)   # split any dataset according to the split in 'split_dir'

.. code:: python

    # for k-fold cross validation

    splitter = KFold(kfold_dir)

    for split in splitter.get_splits(dataset):      # here we can iterate over the splits of the K-Fold
        train_set = split.train_dataset             # a CapsDataset
        val_set = split.val_dataset                 # another CapsDataset
        ...

.. note::
    If you want to split your dataset between a training and a test set, without having to
    manipulate a :py:class:`~clinicadl.splitter.Split` object, you can just create two different
    datasets:

    .. code::

        train_set = CapsDataset("mycaps", data=split_dir / "train.tsv")
        test_set = CapsDataset("mycaps", data=split_dir / "test_baseline.tsv")

    Or use :py:meth:`CapsDataset.subset() <clinicadl.data.datasets.CapsDataset.subset>`:

    .. code ::

        dataset = CapsDataset("mycaps", data="mycaps/participants_sessions.tsv")
        train_set = dataset.subset(split_dir / "train.tsv")
        test_set = dataset.subset(split_dir / "test_baseline.tsv")

    However, if you want to use ClinicaDL's ``Trainer``, you'll have to use the
    :py:class:`~clinicadl.splitter.Split` object.

.. toctree::
    :maxdepth: 1

    single_split
    k_fold