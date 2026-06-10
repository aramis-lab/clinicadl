.. _user_guide_data_splitting:

1.4. Splitting data
===================

To estimate how well a model generalises, you split your data into a **training**
set, a **validation** set, and usually a held-out **test** set. In neuroimaging,
this split deserves special care because datasets are often **longitudinal**: the
same participant is scanned at several sessions.

.. warning::

    If two sessions of the *same participant* end up on different sides of a split —
    one in training, one in validation or test — the model can recognise the
    participant rather than the condition you are studying. This is a classic case of
    :term:`data leakage`, and it leads to over-optimistic performance estimates.

    The rule is therefore to split **by participant**: all the sessions of a
    participant stay together. Every ClinicaDL splitting tool enforces this.

ClinicaDL separates **making** a split (computing it once and saving it to disk) from
**reading** it (applying it to a dataset). Saving the split to disk means the exact
same partition can be reused and shared, which is essential for reproducible
benchmarks.

.. _user_guide_data_splitting_making:

1.4.1. Making a split
---------------------

Two functions create a split from a DataFrame (or a TSV file) listing your
``(participant_id, session_id)`` pairs. Both split **participants** and write the
resulting TSV files to a directory, which they return.

:py:func:`~clinicadl.split.make_split`
    Creates a single train/test (or train/validation) split. ``n_test`` sets the
    size of the held-out set, as a proportion of participants or as an absolute
    number.

:py:func:`~clinicadl.split.make_kfold`
    Creates a :math:`K`-fold partition: ``n_splits`` folds, each used in turn as the
    validation set.

.. code-block:: python

    from clinicadl.split import make_split, make_kfold

    # a single 80/20 split
    split_dir = make_split("bids_directory/metadata.tsv", n_test=0.2)

    # a 5-fold partition
    kfold_dir = make_kfold(split_dir / "training.tsv", n_splits=5)

Both functions support two important options:

``longitudinal``
    By default (``longitudinal=False``), only the **baseline** session of each
    participant is kept in the held-out set, while the training set always keeps all
    sessions. Set ``longitudinal=True`` to keep every session of the held-out
    participants as well.

``stratification``
    Ensures the resulting sets have similar distributions for one or several
    variables (e.g. age, sex). ``make_split`` checks the balance with statistical
    tests and retries until it finds a satisfactory split; ``make_kfold`` stratifies
    on a single categorical variable. A ``seed`` makes the split reproducible.

.. code-block:: python

    split_dir = make_split(
        "bids_directory/metadata.tsv",
        n_test=0.2,
        stratification=["age", "sex"],
        longitudinal=True,
        seed=42,
    )

The output directory contains the train/test TSV files and, when stratification is
used, statistics describing how well the variables are balanced:

.. code-block:: text

    splits/split
    ├── single_split_config.json
    ├── split_categorical_stats.tsv
    ├── split_continuous_stats.tsv
    ├── test_baseline.tsv
    ├── train.tsv
    └── train_baseline.tsv

.. _user_guide_data_splitting_reading:

1.4.2. Reading a split
----------------------

Once a split exists on disk, you apply it to a :py:class:`~clinicadl.data.datasets.Dataset`
with one of two reader objects, depending on how the split was made.

:py:class:`~clinicadl.split.SingleSplit`
    Reads a directory produced by :py:func:`~clinicadl.split.make_split`. Its
    :py:meth:`~clinicadl.split.SingleSplit.get_split` method returns a single
    :py:class:`~clinicadl.split.Split`.

    .. code-block:: python

        from clinicadl.split import SingleSplit
        from clinicadl.data.datasets import BidsDataset
        from clinicadl.io.bids import BidsFileType

        dataset = BidsDataset(
            "bids_directory",
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
            data="bids_directory/metadata.tsv",
        )

        split = SingleSplit(split_dir).get_split(dataset)

:py:class:`~clinicadl.split.KFold`
    Reads a directory produced by :py:func:`~clinicadl.split.make_kfold`. Its
    :py:meth:`~clinicadl.split.KFold.get_splits` method **yields** one
    :py:class:`~clinicadl.split.Split` per fold.

    .. code-block:: python

        from clinicadl.split import KFold

        splitter = KFold(kfold_dir)
        for split in splitter.get_splits(dataset):
            ...   # train and evaluate on this fold

A :py:class:`~clinicadl.split.Split` simply bundles the two resulting datasets:

.. code-block:: python

    >>> split.train_dataset
    <clinicadl.data.datasets.BidsDataset object ...>
    >>> split.val_dataset
    <clinicadl.data.datasets.BidsDataset object ...>

.. note::

    The :py:class:`~clinicadl.split.Split` object also knows how to build the
    :py:class:`~clinicadl.data.dataloader.DataLoader` of each set. We look at
    dataloaders in the :doc:`next section <dataloader>`.

Evaluating on a different view of the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both :py:class:`SingleSplit.get_split <clinicadl.split.SingleSplit.get_split>` and :py:class:`KFold.get_splits <clinicadl.split.KFold.get_splits>`
accept an ``eval_dataset`` argument. This is handy when you want to **train on patches but evaluate on whole images**:
pass the patch dataset as the main argument and the image dataset as ``eval_dataset``.

.. code-block:: python

    from clinicadl.transforms import TransformsHandler, extraction

    train_dataset = BidsDataset(
        "bids_directory",
        file_type=BidsFileType(data_type="anat", suffix="T1w"),
        data="bids_directory/metadata.tsv",
        transforms=TransformsHandler(extraction=extraction.Patch(patch_size=64)),
    )
    eval_dataset = BidsDataset(
        "bids_directory",
        file_type=BidsFileType(data_type="anat", suffix="T1w"),
        data="bids_directory/metadata.tsv",
    )

    split = SingleSplit(split_dir).get_split(train_dataset, eval_dataset=eval_dataset)

.. code-block:: python

    >>> split.train_dataset[0].spatial_shape
    (64, 64, 64)              # patches for training
    >>> split.val_dataset[0].spatial_shape
    (181, 217, 181)          # entire images for evaluation

----

Once your data is divided into training and validation sets, the final step before starting training is to specify how it will be batched.
