.. _user_guide_data_structures:

1.1 Data structures
====================

In neuroimaging, an image rarely travels alone: it comes with a participant and a
session identifier, sometimes with one or several masks (for instance, marking specific anatomical regions),
and with metadata such as the age of the participant or a
diagnosis label. Storing all of these elements within one unified object helps prevent common errors —
for instance applying a spatial transform to an image but forgetting to apply it to its mask.

This is the role of the :py:class:`~clinicadl.data.structures.DataPoint`, the
central data structure of ClinicaDL, and of its child the
:py:class:`~clinicadl.data.structures.Sample`.

The DataPoint
-------------

A :py:class:`~clinicadl.data.structures.DataPoint` gathers an image and any other
relevant information associated with it. It is a subclass of
:py:class:`torchio.Subject` (which itself behaves like a Python ``dict``), so any
TorchIO operation that works on a ``Subject`` also works on a ``DataPoint``.

A ``DataPoint`` always has at least three fields:

- ``image``: the image, as a :py:class:`torchio.ScalarImage`;
- ``participant_id``: the participant id, as a ``str``;
- ``session_id``: the session id, as a ``str``.

.. code-block:: python

    import numpy as np
    import torch
    import torchio as tio
    from clinicadl.data.structures import DataPoint

    datapoint = DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(1, 10, 10, 10), affine=np.eye(4)),
        participant_id="sub-001",
        session_id="ses-M000",
    )

You can pass the ``image`` either as a :py:class:`torchio.ScalarImage` or simply as
a path to a NIfTI file. Any extra keyword argument is stored as an additional
field.

Accessing fields
~~~~~~~~~~~~~~~~~

The three core fields are available as attributes:

.. code-block:: python

    >>> datapoint.session_id
    'ses-M000'

Any other field is accessed with the usual dictionary syntax, which you also use to
add or modify a field:

.. code-block:: python

    >>> datapoint["age"] = 55
    >>> datapoint["age"]
    55

Throughout this guide we use the bundled
:py:class:`~clinicadl.data.structures.examples.Colin27DataPoint`, a ready-to-use
``DataPoint`` wrapping the `Colin 27 average brain <https://www.bic.mni.mcgill.ca/ServicesAtlases/Colin27Highres>`.
It contains a T1 image and a mask named ``"head"``, and requires no external data:

.. code-block:: python

    >>> from clinicadl.data.structures.examples import Colin27DataPoint
    >>> datapoint = Colin27DataPoint()
    >>> datapoint
    Colin27DataPoint(Keys: ('head', 'image', 'participant_id', 'session_id'); images: 2)
    >>> datapoint.participant_id
    'sub-colin'

Images, masks and other fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``DataPoint`` distinguishes three kinds of content: images (stored as
:py:class:`torchio.ScalarImage`), masks (stored as :py:class:`torchio.LabelMap`)
and everything else (metadata). Dedicated helpers let you retrieve each kind:

.. code-block:: python

    >>> datapoint.get_images_dict()
    {'image': ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)}
    >>> datapoint.get_masks_dict()
    {'head': LabelMap(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)}
    >>> datapoint.get_non_images_dict()
    {'participant_id': 'sub-colin', 'session_id': 'ses-M000'}

To add an image or a mask, prefer :py:meth:`~clinicadl.data.structures.DataPoint.add_image`
and :py:meth:`~clinicadl.data.structures.DataPoint.add_mask` over the raw dictionary
assignment: they accept a :py:class:`torchio.ScalarImage`/:py:class:`torchio.LabelMap`,
a path to a NIfTI file, or a 4D :py:class:`torch.Tensor` (in which case the affine
matrix of ``image`` is reused).

.. code-block:: python

    >>> datapoint.add_image(datapoint.image, "image_duplicate")
    >>> datapoint
    Colin27DataPoint(Keys: ('head', 'image', 'participant_id', 'session_id', 'image_duplicate'); images: 3)

To get the raw tensor of an image, use
:py:meth:`~clinicadl.data.structures.DataPoint.get_image_tensor`.

Spatial properties
~~~~~~~~~~~~~~~~~~~

When all the images and masks of a ``DataPoint`` share the same geometry, you can
access it directly through the :py:attr:`~clinicadl.data.structures.DataPoint.shape`
(or :py:attr:`~clinicadl.data.structures.DataPoint.spatial_shape` without the channel
dimension), :py:attr:`~clinicadl.data.structures.DataPoint.spacing` and
:py:attr:`~clinicadl.data.structures.DataPoint.affine` properties. Consistency across
images is checked on access.

.. code-block:: python

    >>> datapoint.shape
    (1, 181, 217, 181)
    >>> datapoint.spatial_shape
    (181, 217, 181)
    >>> datapoint.spacing
    (1.0, 1.0, 1.0)

Finally, since a ``DataPoint`` is a :py:class:`torchio.Subject`, you can visualise
its images with :py:meth:`~clinicadl.data.structures.DataPoint.plot`.

.. tip::

    Because a ``DataPoint`` is a dictionary, you can store anything in it.
    This is what makes it a convenient container to move data
    through the whole ClinicaDL pipeline.

The Sample
----------

A :py:class:`~clinicadl.data.structures.Sample` is a ``DataPoint`` with a few extra
fields. It is the **output of a** :py:class:`~clinicadl.data.datasets.Dataset` (see
:doc:`Reading BIDS datasets <bids>`): when a dataset loads an image — or extracts a
patch or a slice from it — it returns a ``Sample``.

In addition to the ``DataPoint`` fields, a ``Sample`` carries:

- ``file_type``: the :py:class:`~clinicadl.io.bids.BidsFileType` describing the loaded file(s);
- ``image_path``: the path(s) to the loaded image;
- ``sample_type``: the kind of sample, among ``"image"``, ``"patch"`` and ``"slice"``;
- ``sample_position``: the position of the sample in the original image (the index of
  a slice, or the coordinates of a patch), or ``None`` for a whole image.

A bundled :py:class:`~clinicadl.data.structures.examples.Colin27Sample` illustrates it:

.. code-block:: python

    >>> from clinicadl.data.structures.examples import Colin27Sample
    >>> sample = Colin27Sample()
    >>> sample.sample_type
    'image'
    >>> sample.sample_position    # None: a whole image has no position
    >>> sample
    Colin27Sample(Keys: ('head', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 2)

The 2D Sample
-------------

When you work on 2D slices rather than 3D volumes, a dataset returns a
:py:class:`~clinicadl.data.structures.Sample2D`, which is a ``Sample`` with two additional
fields:

- ``slice_direction``: the axis along which slicing was performed — ``0``, ``1``, or ``2``;
- ``squeeze``: whether the slice tensor should be squeezed to two spatial dimensions
  (most ClinicaDL operations work internally with 3D tensors, so the dummy dimension
  is only removed when needed, e.g. right before the data is passed to a 2D neural network).

.. code-block:: python

    >>> from clinicadl.data.structures.examples import Colin27Sample2D
    >>> slice_sample = Colin27Sample2D()
    >>> slice_sample.sample_type
    'slice'
    >>> slice_sample.slice_direction
    1
    >>> slice_sample.sample_position
    108

Note how :py:meth:`~clinicadl.data.structures.Sample2D.get_image_tensor` returns a
squeezed tensor when ``squeeze=True``.

----

Now that you know the objects ClinicaDL uses to carry your data, the next section
shows how a :py:class:`~clinicadl.data.datasets.BidsDataset` produces these
``Samples`` from a :term:`BIDS` directory.
