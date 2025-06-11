.. _capsdataset_outputs:

Outputs of CapsDataset
======================

The output data of a :py:class:`~clinicadl.data.datasets.CapsDataset` is always a :py:class:`~clinicadl.data.structures.DataPoint`,
but depending on the extraction performed, the DataPoint does not contain the same information. 

Image
-----

.. autoclass:: clinicadl.transforms.extraction.image.ImageSample()
    :members: get_tensors

Patch
-----

.. autoclass:: clinicadl.transforms.extraction.patch.PatchSample()
    :members: get_tensors

Slice
-----

.. autoclass:: clinicadl.transforms.extraction.slice.SliceSample()
    :members: get_tensors