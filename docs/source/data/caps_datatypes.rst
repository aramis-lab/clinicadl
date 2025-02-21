.. _caps_datatypes:

CAPS datatypes
==============
.. currentmodule:: clinicadl.data.datatypes

Let's take the following CAPS structure::

    caps
    ├── subjects
    │   ├── sub-000
    │   │   └── ses-M000
    │   │   │   └── pet_linear
    │   │   │   │   ├── sub-000_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.nii.gz
    │   │   │   │   └── sub-000_ses-M000_trc-18FFDG_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_pet.nii.gz
    │   │   │   └── t1_linear
    |   │   │       ├── sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz
    |   │   │       └── sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz
    │   │   └── ...
    │   └── ...


Having a (participant, session) is clearly not enough to precise the data we want to work on.
To precise it, ClinicaDL introduces objects that represent the different types of data. For example,
if you want here to work on PET scans acquired with ``18FFDG`` and preprocessed with Clinica's ``pet-linear``
pipeline with ``pons2`` as a reference region for SUVR computation, you will create the associated object:

.. code-block:: python

    >>> from clinicadl.data.datatypes import PETLinear
    >>> PETLinear(tracer="18FFDG", suvr_reference_region="pons2")
    PETLinear(
        use_uncropped_image=False,
        tracer="18FFDG",
        reconstruction=None,
        suvr_reference_region="pons2",
        modality="pet",
        name="pet-linear",
        file_type=FileType(
            pattern="pet_linear/sub-*_ses-*_trc-18FFDG_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_pet.nii*",
            description="PET images with tracer '18FFDG', registered to MNI152NLin2009cSym space using Clinica's 'pet-linear' pipeline with SUVR reference region 'pons2', and cropped (matrix size 169×208×179, 1 mm isotropic voxels)",
            needed_pipeline="pet-linear",
        ),
    )

Currently, ClinicaDL accepts images preprocessed with ``t1-linear``, ``flair-linear``, ``pet-linear``
and ``dwi-dti`` Clinica pipelines. If you want to work with data that have not been preprocessed with
Clinica, you should use :py:class:`Custom`.

Clinica ``t1-linear``
---------------------
.. autoclass:: T1Linear

Clinica ``flair-linear``
------------------------
.. autoclass:: FlairLinear

Clinica ``pet-linear``
----------------------
.. autoclass:: PETLinear

Clinica ``dwi-dti``
-------------------
.. autoclass:: DWIDTI

Data not preprocessed with Clinica
----------------------------------
.. autoclass:: Custom

