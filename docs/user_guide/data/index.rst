.. _user_guide_data:

1. Manipulating neuroimaging data
=================================

Before training any model, you need to **load, organise and transform** your
neuroimaging data. This is precisely what the data tools of ClinicaDL are for, and
this chapter walks you through them, from the elementary data structures to the
batches that will feed your network.

The chapter follows the natural life cycle of the data in a ClinicaDL experiment:

#. :doc:`Data structures <structures>` — the objects ClinicaDL uses to carry an
   image, its masks and its metadata together: the
   :py:class:`~clinicadl.data.structures.DataPoint` and its child the
   :py:class:`~clinicadl.data.structures.Sample`.
#. :doc:`Reading BIDS datasets <bids>` — how to read a :term:`BIDS` directory with
   a :py:class:`~clinicadl.data.datasets.BidsDataset`, how to speed up loading by
   converting images to tensors, and how to combine several datasets.
#. :doc:`Transforming data <transforms>` — how to extract patches or slices and how
   to apply preprocessing, data augmentation and post-processing with a
   :py:class:`~clinicadl.transforms.TransformsHandler`.
#. :doc:`Splitting data <splitting>` — how to build training, validation and test
   sets without :term:`data leakage`.
#. :doc:`Batching data for training <dataloader>` — how to iterate over a dataset in
   batches with a :py:class:`~clinicadl.data.dataloader.DataLoader`.

.. note::

    Most examples in this chapter are **runnable as-is**: they rely on the bundled
    :py:class:`~clinicadl.data.structures.examples.Colin27DataPoint` and its
    relatives, which wrap the public Colin 27 average brain and require no external
    data. Examples that need a dataset on disk describe the expected :term:`BIDS`
    tree so that you can adapt them to your own data.

.. toctree::
   :maxdepth: 2
   :hidden:

   structures
   bids
   transforms
   splitting
   dataloader
