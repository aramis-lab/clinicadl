.. _data:

Data
====

At the moment, ClinicaDL only works with data organized in a
`CAPS <https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/>`_
structure.

To manipulate data organized in a CAPS structure, ClinicaDL introduces
:py:class:`~clinicadl.data.datasets.CapsDataset`, which is a :py:class:`torch.utils.data.Dataset`
with some specificities.

A CAPS structure can contain different kinds of data (e.g. different modalities or
preprocessings). To define the type of data you want to manipulate, you must pass
to the ``CapsDataset`` a :ref:`CAPS datatype <caps_datatypes>`, which is a
representation of these data. ``CapsDataset`` will use this object to get the
right images in your CAPS structure.

To handle multiple datasets and/or multiple modalities, you may be interested in
:py:class:`~clinicadl.data.datasets.ConcatDataset`, :py:class:`~clinicadl.data.datasets.PairedDataset` or
:py:class:`~clinicadl.data.datasets.UnpairedDataset`.

To transform your data or perform data augmentation, you will use :py:class:`~clinicadl.transforms.Transforms`.
This object aims to gather all the transforms that will be applied to the images
when loaded by ``CapsDataset`` (e.g. preprocessing, augmentation, patch/slice extraction).

Once your dataset has been created, you'll probably want to put it in a :py:class:`torch.utils.data.DataLoader`.
To do this, you can use :py:class:`~clinicadl.data.dataloader.DataLoaderConfig`. This object will help you to
create a DataLoader suited to ClinicaDL.

Finally, before training a Deep Learning model, you will split your data between training, validation and test sets.
To do this, you will manipulate our :ref:`splitting tools <splitter>`.

.. toctree::
    :maxdepth: 1

    datasets/caps_dataset
    datasets/concat_dataset
    datasets/paired_dataset
    datasets/unpaired_dataset
    datatypes
    transforms/transforms
    dataloader
    splitter/splits