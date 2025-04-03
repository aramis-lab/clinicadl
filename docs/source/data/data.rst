.. _data:

Data
====

At the moment, ClinicaDL only works with data organized in a
`CAPS <https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/>`_
structure.

To manipulate data organized in a CAPS structure, ClinicaDL introduces
:py:class:`CapsDataset <clinicadl.data.datasets.CapsDataset>`, which is a :py:class:`torch.utils.data.Dataset`
with some specificities.

A CAPS structure can contain different kinds of data (e.g. different modalities or
preprocessings). To define the type of data you want to manipulate, you must pass
to the ``CapsDataset`` a :ref:`CAPS datatype <caps_datatypes>`, which is a
representation of these data. ``CapsDataset`` will use this object to get the
right images in your CAPS structure.

Finally, the last ClinicaDL object that needs to be introduced here is :py:class:`Transforms <clinicadl.transforms.Transforms>`.
This object aims to gather all the transforms that will be applied to the images
when loaded by ``CapsDataset`` (e.g. preprocessing, augmentation, patch/slice extraction).

.. toctree::
    :maxdepth: 1

    caps_dataset
    concat_dataset
    paired_dataset
    unpaired_dataset
    caps_datatypes
    ../transforms/transforms