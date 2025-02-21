.. _capsdataset_outputs:

Outputs of CapsDataset
======================

Depending on the extraction performed, :py:class:`CapsDataset <clinicadl.data.datasets.CapsDataset>`
does not return the same object.

Image
-----

.. autoclass:: clinicadl.transforms.extraction.image.ImageSample

Patch
-----

.. autoclass:: clinicadl.transforms.extraction.patch.PatchSample

Slice
-----

.. autoclass:: clinicadl.transforms.extraction.slice.SliceSample