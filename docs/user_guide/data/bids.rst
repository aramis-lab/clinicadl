.. _user_guide_data_bids:

1.2. Reading BIDS datasets
==========================

ClinicaDL reads neuroimaging datasets organised in the :term:`BIDS` format (Brain
Imaging Data Structure), as well as :term:`BIDS derivatives <BIDS derivative>` and
:term:`CAPS` directories produced by :clinica:`Clinica <>`. Reading a dataset
involves three objects:

- a :py:class:`~clinicadl.io.bids.Bids`, which knows how to navigate a BIDS-like
  directory;
- a :py:class:`~clinicadl.io.bids.BidsFileType`, which describes which files you
  want;
- a :py:class:`~clinicadl.data.datasets.BidsDataset`, which ties the two together
  and loads the selected files into :py:class:`Samples <clinicadl.data.structures.Sample>`.

Navigating a BIDS directory
---------------------------

A :py:class:`~clinicadl.io.bids.Bids` is created from the path to a BIDS-like
directory. The directory must contain the mandatory
:bids:`dataset_description.json <modality-agnostic-files/dataset-description.html#dataset_descriptionjson>`
file, whose ``"DatasetType"`` key (``"raw"``, ``"derivative"`` or ``"study"``) tells
ClinicaDL how the directory is organised.

.. code-block:: python

    from clinicadl.io.bids import Bids

    bids = Bids("bids")

Most of the time you will not call the methods of ``Bids`` directly — the
``BidsDataset`` does it for you. They are however useful for inspecting a dataset:
:py:meth:`~clinicadl.io.bids.Bids.get_all_participants_sessions` lists every
``(participant, session)`` pair in the directory, while
:py:meth:`~clinicadl.io.bids.Bids.get_path` and
:py:meth:`~clinicadl.io.bids.Bids.has_file_type` locate a specific file.

Describing the files to load
----------------------------

A :py:class:`~clinicadl.io.bids.BidsFileType` defines the files you are interested
in. It is expressed in the vocabulary of the BIDS specification — a
:bids:`suffix <common-principles.html#filenames>`, a
:bids:`data type <common-principles.html#definitions>` (the folder, e.g.
``"anat"``), a file extension, and the :bids:`entities <common-principles.html#entities>`
the files must (or must not) contain. `Regular expressions <https://www.w3schools.com/python/python_regex.asp>`_
are accepted everywhere.

For example, to select all the isotropic T1-weighted images registered to the MNI space:

.. code-block:: python

    from clinicadl.io.bids import BidsFileType

    file_type = BidsFileType(
        data_type="anat",
        suffix="T1w",
        extension=".nii.gz",
        with_entities={"space": r"MNI152.*", "res": "1x1x1"},
    )

Clinica preprocessing pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your data were preprocessed with :clinica:`Clinica <>`, you do not have to write
the file type by hand: ClinicaDL ships ready-made ``BidsFileType`` subclasses that
match the output of the most common Clinica pipelines.

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - File type
      - Selects
    * - :py:class:`~clinicadl.io.bids.T1Linear`
      - T1-weighted images from :clinica:`t1-linear <Pipelines/T1_Linear/>`
    * - :py:class:`~clinicadl.io.bids.FlairLinear`
      - FLAIR images from :clinica:`flair-linear <Pipelines/FLAIR_Linear/>`
    * - :py:class:`~clinicadl.io.bids.PetLinear`
      - PET images from :clinica:`pet-linear <Pipelines/PET_Linear/>`
    * - :py:class:`~clinicadl.io.bids.DwiDti`
      - DTI-based measures from :clinica:`dwi-dti <Pipelines/DWI_DTI/>`

.. code-block:: python

    from clinicadl.io.bids import T1Linear, PetLinear

    t1 = T1Linear()
    pet = PetLinear(tracer="18FFDG", suvr_reference_region="pons")

The BidsDataset
---------------

A :py:class:`~clinicadl.data.datasets.BidsDataset` is the object you will use most.
It reads a BIDS directory, loads the files described by a ``BidsFileType``, and returns
one :py:class:`~clinicadl.data.structures.Sample` per element of the data.

Consider a dataset whose metadata are stored in a TSV file:

.. code-block:: text

    bids
    ├── dataset_description.json
    ├── metadata.tsv
    ├── sub-001
    │   ├── ses-M000
    │   │   └── anat
    │   │       └── sub-001_ses-M000_T1w.nii.gz
    │   ...
    ...

    # metadata.tsv
    participant_id  session_id   age   diagnosis
    sub-001         ses-M000     55.0  control
    sub-002         ses-M000     62.0  patient
    ...

The simplest dataset selects the T1 images of the participants listed in the TSV
file:

.. code-block:: python

    from clinicadl.data.datasets import BidsDataset
    from clinicadl.io.bids import BidsFileType

    dataset = BidsDataset(
        bids="bids",
        file_type=BidsFileType(data_type="anat", suffix="T1w"),
        data="bids/metadata.tsv",
    )

.. code-block:: python

    >>> len(dataset)
    50  # one sample per line of metadata.tsv
    >>> dataset[0]
    Sample(Keys: ('file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 1)
    >>> dataset[0].participant_id, dataset[0].session_id
    ('sub-001', 'ses-M000')

A few arguments shape what a ``BidsDataset`` contains and will output:

``data``
    A :py:class:`pandas.DataFrame` (or a path to a TSV file) listing the
    ``(participant_id, session_id)`` pairs to keep, plus any extra columns. If
    omitted, **all** the pairs that have the requested ``file_type`` are used.

``columns``
    The columns of ``data`` to carry into each ``Sample``. You can pass a list of
    column names, or a dictionary mapping a column name to a function applied to the
    whole column — handy, for instance, to encode string labels as integers:

    .. code-block:: python

        import pandas as pd

        def encode_diagnosis(column: pd.Series) -> pd.Series:
            return column.map({"control": 0, "patient": 1})

        dataset = BidsDataset(
            bids="bids",
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
            data="bids/metadata.tsv",
            columns={"age": None, "diagnosis": encode_diagnosis},
        )

    .. code-block:: python

        >>> dataset[0]["diagnosis"]
        0

``masks``
    Masks to load alongside each image, passed as a dictionary. The keys become the
    mask names in the ``Sample``; the values describe where to find each mask — a
    single shared NIfTI file or a :py:class:`~clinicadl.io.bids.BidsFileType` (for a
    participant- and session-specific mask in the same BIDS).

    .. code-block:: text

        bids
        ├── dataset_description.json
        ├── metadata.tsv
        ├── sub-001
        ...
        └── derivatives
            ├── registration
            │   ├── space-MNI152NLin2009cSym_mask.nii.gz
            │   ...
            └── masks
                ├── dataset_description.json
                ├── sub-001
                │   ├── ses-M000
                │   │   └── anat
                │   │       └── sub-001_ses-M000_label-brain_mask.nii.gz
                │   ...
                ...

    .. code-block::

        dataset = BidsDataset(
            bids="bids",
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
            masks={
                "brain": (                                                                   # subject- and session-specific mask that is in another BIDS
                    "bids/derivatives/masks",
                    BidsFileType(
                        data_type="anat", suffix="mask", with_entities={"label": "brain"}
                    ),
                ),
                "mni": "bids/derivatives/registration/space-MNI152NLin2009cSym_mask.nii.gz",  # same mask for all (subject, session)
            },
        )

``transforms``
    A :py:class:`~clinicadl.transforms.TransformsHandler` describing the transforms
    to apply and whether you work on entire images, patches or slices. This is the
    subject of the :doc:`next section <transforms>`. For example, switching to
    patches changes the number of samples in the dataset:

    .. code-block:: python

        from clinicadl.transforms import TransformsHandler, extraction

        dataset = BidsDataset(
            bids="bids",
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
            data="bids/metadata.tsv",
            transforms=TransformsHandler(extraction=extraction.Patch(patch_size=64)),
        )

    .. code-block:: python

        >>> dataset[0].spatial_shape
        (64, 64, 64)    # a patch, not the full image
        >>> len(dataset)
        1800

.. note::

    The **length** of a ``BidsDataset`` is the number of images **times** the number
    of samples extracted per image. With 50 images and 36 patches each, the dataset
    has :math:`50\times36 = 1800` samples.

A ``BidsDataset`` exposes its metadata as a DataFrame
through :py:attr:`~clinicadl.data.datasets.Dataset.df`, and can be restricted to a
subset of ``(participant, session)`` pairs with
:py:meth:`~clinicadl.data.datasets.Dataset.subset`.

.. _user_guide_data_bids_tensors:

1.2.1. Converting NIfTI images to tensors
-----------------------------------------

Opening a NIfTI file is comparatively slow. When you iterate over a dataset many
times — which is exactly what is done during training phase — it pays off to convert your images to
PyTorch tensors once and read from the ``.pt`` files afterwards. This is what
:py:meth:`BidsDataset.to_tensors <clinicadl.data.datasets.BidsDataset.to_tensors>`
does:

.. code-block:: python

    dataset = BidsDataset(
        bids="bids",
        file_type=BidsFileType(data_type="anat", suffix="T1w"),
    )

    tensor_dataset = dataset.to_tensors(conversion_name="T1")

The tensors are written to a :term:`BIDS derivative` named `"tensors"``, together with a ``.json``
file describing the conversion. ``to_tensors`` returns a
:py:class:`~clinicadl.data.datasets.TensorDataset` that you can use exactly like the
original ``BidsDataset`` — only faster to load.

You can also **save the transformed images** (``save_transforms=True``) so that the
``image_transforms`` of your :py:class:`~clinicadl.transforms.TransformsHandler` are
applied once and for all during the conversion instead of on every load. Conversion
can be parallelised with ``n_proc``.

To reopen a previously converted dataset, point a
:py:class:`~clinicadl.data.datasets.TensorDataset` at the description ``.json`` file:

.. code-block:: python

    from clinicadl.data.datasets import TensorDataset

    tensor_dataset = TensorDataset(
        description_json="bids/derivatives/tensors/src-T1w_conv-T1_description.json",
        data="bids/metadata.tsv",
    )

.. warning::

    If you saved the transformed images during the conversion, do not apply the same
    ``image_transforms`` again when re-reading the data: the ``image_transforms`` of
    the ``TransformsHandler`` you pass to ``TensorDataset`` should usually be empty.

.. _user_guide_data_bids_joining:

1.2.2. Joining multiple datasets
--------------------------------

You may need to combine several datasets — images coming from different cohorts,
or different modalities of the same participants. ClinicaDL offers three ways to do
so.

Concatenating
~~~~~~~~~~~~~~

:py:class:`~clinicadl.data.datasets.ConcatDataset` concatenates datasets end to end. The
length of the result is the sum of the lengths of its parts. Use it to gather images
of the **same nature** coming from **different sources**.

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
``(participant, session)`` pairs and the same number of samples per image.

.. code-block:: python

    from clinicadl.data.datasets import BidsDataset, PairedDataset
    from clinicadl.io.bids import BidsFileType

    bids_t1 = BidsDataset("bids", file_type=BidsFileType(data_type="anat", suffix="T1w"))
    bids_pet = BidsDataset("bids", file_type=BidsFileType(data_type="pet", suffix="pet"))

    multimodal_dataset = PairedDataset([bids_t1, bids_pet])

.. code-block:: python

    >>> sample = multimodal_dataset[0]
    >>> len(sample)            # one Sample per modality
    2

Stacking
~~~~~~~~

:py:class:`~clinicadl.data.datasets.UnpairedDataset` also returns a tuple of samples,
but associates the datasets **randomly** rather than through a fixed mapping. The
datasets need not share their ``(participant, session)`` pairs — this is useful, for
example, to feed a generative model with images that should *not* be paired. The
random association can be re-drawn for each epoch with
:py:meth:`~clinicadl.data.datasets.UnpairedDataset.set_epoch`, and the ``oversample``
argument controls how datasets of different sizes are reconciled.

1.2.3. Non-BIDS dataset?
------------------------

If for some reasons, any of the previous dataset class is able to read your data,
you can still write your own dataset by inheriting from :py:class:`clinicadl.data.datasets.Dataset`.

----

You now know how to read your data and assemble it into datasets. The next section
covers the ``transforms`` argument we have only mentioned so far: how to extract
patches and slices, and how to preprocess and augment your images.
