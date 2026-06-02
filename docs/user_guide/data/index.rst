.. _user_guide_data:

1. Manipulating neuroimaging data
=================================

Before training any model, you need to **load**, **organise** and **transform** your
neuroimaging data. This is precisely what the data tools of ClinicaDL are for, and
this chapter walks you through them, from the elementary data structures to the
batches that will feed your network.

#. :doc:`Data structures <structures>` — the objects ClinicaDL uses to carry an
   image, its masks and its metadata together.
#. :doc:`Reading BIDS datasets <bids>` — how to read a :term:`BIDS` directory,
   how to speed up loading by converting images to tensors, and how to combine several datasets.
#. :doc:`Transforming data <transforms>` — how to extract patches or slices from images, and how
   to apply pre-processing, data augmentation and post-processing.
#. :doc:`Splitting data <splitting>` — how to build training, validation and test
   sets without :term:`data leakage`.
#. :doc:`Batching data for training <dataloader>` — how to iterate over a dataset in
   batches.

.. toctree::
   :maxdepth: 2
   :hidden:

   structures
   bids
   transforms
   splitting
   dataloader
