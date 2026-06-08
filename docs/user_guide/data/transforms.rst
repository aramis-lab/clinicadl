.. _user_guide_data_transforms:

1.3 Transforming data
=====================

Raw neuroimaging data is rarely fed to a network as-is: it is normalised, possibly
cut into patches or slices, resized to fit the network, and — during training —
augmented. In ClinicaDL, this whole pipeline is described by a single object, the
:py:class:`~clinicadl.transforms.TransformsHandler`, which you pass to a dataset
through its ``transforms`` argument.

The TransformsHandler
---------------------

A :py:class:`~clinicadl.transforms.TransformsHandler` organises transforms into
**four** stages, applied in this order:

#. ``image_transforms`` — transforms applied to the **whole image, before
   extraction**. This is where normalisation belongs, so that statistics are
   computed on the full image and not on a single patch or slice.
#. ``extraction`` — what you work on: the whole image, patches, or slices.
#. ``sample_transforms`` — transforms applied to a sample (an image, a patch or a slice),
   **after extraction**. This is typically where you resize a sample to fit your
   network.
#. ``augmentations`` — transforms applied last, **only during training**.

.. code-block:: python

    from clinicadl.transforms import TransformsHandler, extraction
    import torchio as tio

    transforms = TransformsHandler(
        image_transforms=[tio.ZNormalization()],
        extraction=extraction.Patch(patch_size=64),
        sample_transforms=[tio.CropOrPad(64)],
        augmentations=[tio.RandomFlip()],
    )

``image_transforms``, ``sample_transforms`` and ``augmentations`` are **sequences**:
the handler composes them in order, so the order matters. A dataset using these
transforms automatically applies the augmentations only when it is in training mode
(see :py:meth:`Dataset.train <clinicadl.data.datasets.Dataset.train>` /
:py:meth:`Dataset.eval <clinicadl.data.datasets.Dataset.eval>`).

.. _user_guide_data_transforms_extraction:

1.3.1 Patches and slices
------------------------

The ``extraction`` decides what a single element of the dataset is. ClinicaDL
provides three extractions, in :py:mod:`clinicadl.transforms.extraction`:

:py:class:`~clinicadl.transforms.extraction.Image`
    No extraction: each sample is a whole image. This is the default.

:py:class:`~clinicadl.transforms.extraction.Patch`
    Each sample is a 3D patch, obtained with a sliding window. The main argument is
    ``patch_size`` (a single value, or one per spatial dimension); ``overlap``,
    ``pad_mode`` and ``pad_value`` control how patches are tiled across the image.

:py:class:`~clinicadl.transforms.extraction.Slice`
    Each sample is a 2D slice taken along the axis specified via ``slice_direction``. Which slices to keep can be chosen with ``slices``,
    ``discarded_slices``, ``borders`` or ``tsv_path``.

The extraction also determines how many samples an image yields, which you can check
on a single :py:class:`~clinicadl.data.structures.DataPoint`:

.. code-block:: python

    from clinicadl.transforms.extraction import Patch
    from clinicadl.data.structures.examples import Colin27DataPoint

    data = Colin27DataPoint()
    patch = Patch(patch_size=64)

.. code-block:: python

    >>> data.spatial_shape
    (181, 217, 181)
    >>> patch.num_samples_per_image(data)
    36
    >>> patch(data, sample_index=0).spatial_shape
    (64, 64, 64)

Slicing works the same way:

.. code-block:: python

    from clinicadl.transforms.extraction import Slice

    slices = Slice(borders=10, slice_direction=1)

.. code-block:: python

    >>> slices.num_samples_per_image(data)
    197     # 217 coronal slices minus 2 × 10 borders
    >>> slices(data, sample_index=0).spatial_shape
    (181, 1, 181)

Remember that, because each image now yields several samples, the **length of a
dataset** is the number of images times the number of samples per image (see
:doc:`Reading BIDS datasets <bids>`).

.. _user_guide_data_transforms_pipeline:

1.3.2 Preprocessing, augmentation and post-processing
-----------------------------------------------------

ClinicaDL works with **any callable that takes a**
:py:class:`~clinicadl.data.structures.DataPoint` **and returns a** ``DataPoint``.
This means you can use, as a transform:

- any :torchio:`TorchIO transform <transforms/transforms.html>` — they operate on a
  :py:class:`torchio.Subject`, of which a ``DataPoint`` is a subclass;
- any of your own functions following the same signature;
- the two transforms ClinicaDL ships in :py:mod:`clinicadl.transforms`,
  :py:class:`~clinicadl.transforms.Format` and
  :py:class:`~clinicadl.transforms.MergeFields`.

For instance, a simple preprocessing-and-augmentation pipeline built entirely from
raw TorchIO transforms:

.. code-block:: python

    import torchio as tio
    from clinicadl.transforms import TransformsHandler, extraction

    transforms = TransformsHandler(
        extraction=extraction.Image(),
        image_transforms=[
            tio.ToCanonical(),         # reorient to RAS+
            tio.ZNormalization(),      # intensity normalisation on the whole image
        ],
        augmentations=[
            tio.RandomFlip(axes=("LR",)),
            tio.RandomAffine(degrees=10),
        ],
    )

A custom transform is just a function. Here is one that adds a binary head mask
derived from the image:

.. code-block:: python

    import torch
    from clinicadl.data.structures import DataPoint
    from clinicadl.data.structures.examples import Colin27DataPoint

    def add_foreground_mask(datapoint: DataPoint) -> DataPoint:
        mask = (datapoint.get_image_tensor("image") > 0).to(torch.int)
        datapoint.add_mask(mask, "foreground")
        return datapoint

    data = Colin27DataPoint()

.. code-block:: python

    >>> add_foreground_mask(data)
    Colin27DataPoint(Keys: ('head', 'image', 'participant_id', 'session_id', 'foreground'); images: 3)

However, it is advised to inherit from :py:class:`torchio.Transform` to create
your own transforms.

.. admonition:: Where do the transforms apply?
    :class: tip

    Put **normalisation** in ``image_transforms`` (computed on the whole image),
    **resizing to the network input size** in ``sample_transforms`` (after a patch or
    slice has been extracted), and **data augmentation** in ``augmentations`` (active
    only during training).

.. note::

    In this chapter we deliberately use **raw transforms** — TorchIO transforms and
    plain functions. ClinicaDL also offers *configuration classes* that wrap many of
    these transforms in a serialisable form, for better reproducibility. Those are
    introduced later, in :doc:`Chapter 3 <../reproducibility/index>`.

----

Your data is now read and transformed. Before training, you need to separate it into
training, validation and test sets — the topic of the :doc:`next section <splitting>`.
