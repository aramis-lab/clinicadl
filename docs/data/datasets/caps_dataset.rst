.. _caps_dataset:

CapsDataset
===========

``CapsDataset`` is the object that you will always use to
manipulate your neuroimaging data stored in a `CAPS <https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/>`_
structure.

As it inherits from PyTorch's :py:class:`Dataset <torch.utils.data.Dataset>`,
it is an iterable, whose length can be accessed via ``len(dataset)``, and whose elements can be
accessed with their indices: ``dataset[i]``.

The additional features of ``CapsDataset`` are described in the documentation below, but there is
a specificity the user should be aware of: **CapsDataset only manipulates PyTorch tensors**. So a
preliminary step is to **convert your NIfTI files to tensors**. The method ``to_tensors`` is here to
help you:

.. code-block:: python

    >>> from clinicadl.data.datasets import CapsDataset
    >>> from clinicadl.data.datatypes import PETLinear
    >>> dataset = CapsDataset(
            caps_directory="mycaps",
            preprocessing=PETLinear(
                tracer="18FAV45", use_uncropped_image=True, suvr_reference_region="pons2"
            ),
            data="mycaps/pet_data.tsv",
            label="seg",
            masks=["brain", "leftHippocampus.nii.gz"],
        ) 
    >>> dataset.to_tensors("pet_conversion")

Your CAPS structure will now look like this::

    mycaps
    ├── masks
    │   ├── leftHippocampus.nii.gz
    │   └── tensors
    │       └── leftHippocampus.pt
    ├── pet_data.tsv
    ├── subjects
    │   ├── sub-000
    │   │   └── ses-M000
    │   │   │   └── pet_linear
    │   │   │   │   ├── sub-000_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_brain.nii.gz
    │   │   │   │   ├── sub-000_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz
    │   │   │   │   ├── sub-000_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_seg.nii.gz
    │   │   │   │   └── tensors
    │   │   │   │       └── sub-000_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.pt
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── tensor_conversion
        └── pet_conversion.json

New ``tensors`` folders has been added to contain the PyTorch tensors.
A file like ``sub-000_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.pt`` contain the image as
a tensor, as well as the individual masks associated to the image (the masks from ``*seg.nii.gz`` and ``*brain.nii.gz``
here). Notice that masks common to all images (stored in ``masks``) have also been converted.

All the useful information (what contains the ``.pt`` files, which transforms have been applied, etc.) on the tensor conversion
is stored in ``tensor_conversion/pet_conversion.json``. This file will be particularly useful if you don't want to make the conversion
again the next time you will instantiate your ``CapsDataset``. In this case, use the ``read_tensor_conversion`` method:

.. code-block:: python

    >>> dataset.read_tensor_conversion("pet_conversion")

.. note::
    ``CapsDataset`` will compare the content of ``tensor_conversion/pet_conversion.json`` with its current state to
    be sure that data in ``.pt`` files are indeed the data you want to manipulate. For example, if you ran
    the tensor conversion with certain transforms but the current ``CapsDataset`` has been created with different
    transforms, this will raise an error.

.. autoclass:: clinicadl.data.datasets.CapsDataset
    :members:
    :exclude-members: converted

See also: 
    - :ref:`concat`
    - :ref:`paired`
    - :ref:`unpaired`