.. _user_guide_data_dataloader:

1.5 Batching data for training
==============================

Training a network means feeding it data in **batches**. ClinicaDL provides a
:py:class:`~clinicadl.data.dataloader.DataLoader` that iterates over a
:py:class:`~clinicadl.data.datasets.Dataset` and groups its samples into a
:py:class:`~clinicadl.data.dataloader.Batch`, ready to be turned into tensors.

The DataLoader
--------------

:py:class:`~clinicadl.data.dataloader.DataLoader` is a subclass of
:py:class:`torch.utils.data.DataLoader`, so it behaves like the PyTorch dataloader
you may already know — with the same parameters (``batch_size``, ``shuffle``,
``num_workers``, ``pin_memory``, …).

.. important::

    Only the ClinicaDL ``DataLoader`` is guaranteed to work with ClinicaDL datasets.
    Prefer it over the raw PyTorch one.

.. code-block:: python

    from clinicadl.data.datasets import BidsDataset
    from clinicadl.io.bids import BidsFileType
    from clinicadl.data.dataloader import DataLoader

    dataset = BidsDataset(
        "bids",
        file_type=BidsFileType(data_type="anat", suffix="T1w"),
        data="bids/metadata.tsv",
    )
    loader = DataLoader(dataset, batch_size=3, shuffle=True)

.. code-block:: python

    >>> batch = next(iter(loader))
    >>> len(batch)
    3

Beyond the standard PyTorch arguments, the ``DataLoader`` adds ``sampling_weights``:
the name of a column of the dataset's :py:attr:`~clinicadl.data.datasets.Dataset.df`
whose values are used as sampling probabilities. This is convenient to oversample
under-represented classes.

The Batch
---------

A :py:class:`~clinicadl.data.dataloader.Batch` is a list of
:py:class:`DataPoints <clinicadl.data.structures.DataPoint>` with a few extra
conveniences. The most useful is
:py:meth:`~clinicadl.data.dataloader.Batch.get_field`, which gathers one field
across all the samples and returns it as a batch-first :py:class:`torch.Tensor` when
possible (and a plain list otherwise):

.. code-block:: python

    from clinicadl.data.structures.examples import Colin27DataPoint
    from clinicadl.data.dataloader import Batch

    batch = Batch([Colin27DataPoint(), Colin27DataPoint()])

.. code-block:: python

    >>> batch.get_field("image").shape
    torch.Size([2, 1, 181, 217, 181])
    >>> batch.get_field("participant")
    ['sub-colin', 'sub-colin']

A ``Batch`` can also move its tensors to a device or memory format via
:py:meth:`~clinicadl.data.dataloader.Batch.to`, and accept new fields produced by a
network via :py:meth:`~clinicadl.data.dataloader.Batch.add_field`,
:py:meth:`~clinicadl.data.dataloader.Batch.add_images` and
:py:meth:`~clinicadl.data.dataloader.Batch.add_masks` — handy to store a model's
output back next to its input.

Collating: from samples to batches
-----------------------------------

How individual samples are assembled into a ``Batch`` is decided by a **collate
function**, passed through the ``collate_fn`` argument and described by the abstract
:py:class:`~clinicadl.data.dataloader.CollateFn`. ClinicaDL chooses a sensible
default, so you usually do not need to set it:

- :py:class:`~clinicadl.data.dataloader.ToBatchCollate` — the default when each
  dataset element is a single :py:class:`~clinicadl.data.structures.Sample`. It
  produces a single ``Batch``.
- :py:class:`~clinicadl.data.dataloader.ToBatchesCollate` — the default when each
  element is a *tuple* of samples, as returned by a
  :py:class:`~clinicadl.data.datasets.PairedDataset` or
  :py:class:`~clinicadl.data.datasets.UnpairedDataset` (see
  :ref:`joining datasets <user_guide_data_bids_joining>`). It produces one ``Batch``
  per dataset.
- :py:class:`~clinicadl.data.dataloader.MergeBatchesCollate` — to merge such a tuple
  into a single ``Batch`` instead.

So, with a :py:class:`~clinicadl.data.datasets.PairedDataset`, the loader returns a
tuple of batches by default:

.. code-block:: python

    from clinicadl.data.datasets import PairedDataset

    paired = PairedDataset([dataset, dataset])
    loader = DataLoader(paired, batch_size=3, shuffle=False)

.. code-block:: python

    >>> batch_t1, batch_pet = next(iter(loader))     # one Batch per modality

To define your own collating behaviour, subclass
:py:class:`~clinicadl.data.dataloader.CollateFn` and implement its ``__call__``
method.

Building loaders from a split
-----------------------------

In practice you build one loader for the training set and one for the validation
set. The :py:class:`~clinicadl.split.Split` returned by a splitter (see
:doc:`Splitting data <splitting>`) does this for you, with sensible defaults
(shuffling on for training, off for validation):

.. code-block:: python

    split.build_train_loader(batch_size=8)
    split.build_val_loader(batch_size=8)

    train_loader = split.train_loader
    val_loader = split.val_loader

----

This closes Chapter 1: you can now load your data, transform it, split it without
leakage, and iterate over it in batches. The :doc:`next chapter <../workflow/index>`
puts these batches to use, building and training a model.
