.. _user_guide_data_bids_tensors:

1.2.2. Reading neuroimaging datasets: advanced tips
---------------------------------------------------

If you are familiar with :py:class:`~clinicadl.data.datasets.BidsDataset` (see :doc:`previous section <bids>`),
you may find the following tools useful for speeding up data loading, joining multiple datasets, or reading non-BIDS-compliant
datasets.

1.2.2.1. Converting NIfTI images to tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Opening a :term:`NIfTI` file is comparatively slow. When you iterate over a dataset many
times, which is exactly what is done during training phase, it pays off to convert your images to
PyTorch tensors once and read from the ``.pt`` files afterwards. This is what
:py:meth:`BidsDataset.to_tensors <clinicadl.data.datasets.BidsDataset.to_tensors>`
does:

.. code-block:: python

    dataset = BidsDataset(
        bids="bids_directory",
        file_type=BidsFileType(data_type="anat", suffix="T1w"),
    )

    tensor_dataset = dataset.to_tensors(conversion_name="T1")

The tensors are written to a :term:`BIDS derivative` named `"tensors"``, together with a ``.json``
file describing the conversion. ``to_tensors`` returns a
:py:class:`~clinicadl.data.datasets.TensorDataset` that you can use exactly like the
original ``BidsDataset``, only faster to load.

You can also **save the transformed images** (``save_transforms=True``) so that the
``image_transforms`` of your :py:class:`~clinicadl.transforms.TransformsHandler` are
applied once during the conversion instead of on every load. Conversion
can be parallelised with ``n_proc``.

To reopen a previously converted dataset, point a
:py:class:`~clinicadl.data.datasets.TensorDataset` at the description ``.json`` file:

.. code-block:: python

    from clinicadl.data.datasets import TensorDataset

    tensor_dataset = TensorDataset(
        description_json="bids_directory/derivatives/tensors/src-T1w_conv-T1_description.json",
        data="bids_directory/metadata.tsv",
    )

.. warning::

    If you saved the transformed images during the conversion, do not apply the same
    ``image_transforms`` again when re-reading the data: the ``image_transforms`` of
    the ``TransformsHandler`` you pass to ``TensorDataset`` should usually be empty.

.. _user_guide_data_bids_joining:

1.2.2.2. Joining multiple datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may need to combine several datasets, for instance images coming from different cohorts,
or different modalities for the same participants. ClinicaDL offers three ways to do
so.

Concatenating
~~~~~~~~~~~~~~

:py:class:`~clinicadl.data.datasets.ConcatDataset` concatenates datasets end to end. The
length of the result is the sum of the lengths of its parts. Use it for instance to gather images
coming from **different cohorts** (i.e. with different participants).

.. code-block:: python

    from clinicadl.data.datasets import BidsDataset, ConcatDataset
    from clinicadl.io.bids import BidsFileType

    bids_1 = BidsDataset("bids_1", file_type=BidsFileType(data_type="pet", suffix="pet"))
    bids_2 = BidsDataset("bids_2", file_type=BidsFileType(data_type="pet", suffix="pet"))

    full_dataset = ConcatDataset([bids_1, bids_2])

.. code-block:: python

    >>> len(bids_1), len(bids_2), len(full_dataset)
    (4, 8, 12)

Pairing
~~~~~~~

:py:class:`~clinicadl.data.datasets.PairedDataset` associates datasets through a
**unique mapping** keyed by the ``(participant, session)`` pairs. It is the tool for
**multimodal** data: each sample becomes a tuple holding the corresponding image from
each dataset. All datasets must therefore contain exactly the same
``(participant, session)`` pairs and the same number of samples per image. Consequently,
``PairedDataset`` does not support participants who are missing a modality.


.. code-block:: python

    from clinicadl.data.datasets import BidsDataset, PairedDataset
    from clinicadl.io.bids import BidsFileType

    bids_t1 = BidsDataset("bids_directory", file_type=BidsFileType(data_type="anat", suffix="T1w"))
    bids_pet = BidsDataset("bids_directory", file_type=BidsFileType(data_type="pet", suffix="pet"))

    multimodal_dataset = PairedDataset([bids_t1, bids_pet])

.. code-block:: python

    >>> sample = multimodal_dataset[0]
    >>> len(sample)            # one Sample per modality
    2

Stacking
~~~~~~~~

:py:class:`~clinicadl.data.datasets.UnpairedDataset` also returns a tuple of samples,
but associates the datasets **randomly** rather than through a fixed mapping. The
datasets need not share their ``(participant, session)`` pairs. This is useful, for
example, to feed a generative model with images that should *not* be paired. The
random association can be re-drawn for each epoch with
:py:meth:`~clinicadl.data.datasets.UnpairedDataset.set_epoch`, and the ``oversample``
argument controls how datasets of different sizes are reconciled.

1.2.2.3. Non-BIDS dataset?
^^^^^^^^^^^^^^^^^^^^^^^^^^

If, for some reasons, none of the previous dataset classes is able to read your data,
you can still create your own dataset by inheriting from :py:class:`clinicadl.data.datasets.Dataset`.

----

You now know how to read your data and assemble it into datasets. The next section
covers the ``transforms`` argument we have only mentioned so far: how to extract
patches and slices, and how to preprocess and augment your images.