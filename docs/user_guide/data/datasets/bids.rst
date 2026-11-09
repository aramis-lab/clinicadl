1.2.1. Reading BIDS datasets
----------------------------

ClinicaDL reads neuroimaging datasets organised in the :term:`BIDS` format, as well as :term:`BIDS derivatives <BIDS derivative>`, and
:term:`CAPS` directories produced by :clinica:`Clinica <>`.

Reading a BIDS dataset involves three objects:

- a :py:class:`~clinicadl.io.bids.Bids`, which knows how to navigate a BIDS-like
  directory;
- a :py:class:`~clinicadl.io.bids.BidsFileType`, which describes which files you
  want;
- a :py:class:`~clinicadl.data.datasets.BidsDataset`, which ties the two together
  and loads the selected files into :py:class:`Samples <clinicadl.data.structures.Sample>`.

1.2.1.1. Navigating a BIDS directory: ``Bids``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :py:class:`~clinicadl.io.bids.Bids` is created from the path to a BIDS-like
directory. The directory must contain the mandatory
:bids:`dataset_description.json <modality-agnostic-files/dataset-description.html#dataset_descriptionjson>`
file, whose ``"DatasetType"`` key (``"raw"``, ``"derivative"`` or ``"study"``) tells
ClinicaDL how the directory is organised.

This self-contained example creates toy metadata and empty participant/session
directories so you can try directory navigation without downloading any scans.
It is not a complete BIDS dataset and cannot be used to load images or train a
model.

.. code-block:: python

    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from clinicadl.io.bids import Bids

    with TemporaryDirectory() as tmp_dir:
        bids_directory = Path(tmp_dir)
        description = {
            "Name": "Toy directory navigation example",
            "BIDSVersion": "1.10.0",
            "DatasetType": "raw",
        }
        (bids_directory / "dataset_description.json").write_text(
            json.dumps(description), encoding="utf-8"
        )
        (bids_directory / "sub-01" / "ses-M000").mkdir(parents=True)

        bids = Bids(bids_directory)
        print(sorted(bids.get_all_participants_sessions()))

This prints ``[('sub-01', 'ses-M000')]``. Keep operations on this toy directory
inside the ``with`` block: its contents are removed automatically when the block
ends.

Most of the time you will not call the methods of ``Bids`` directly, the
``BidsDataset`` will do it for you. They are however useful for inspecting a dataset:
:py:meth:`~clinicadl.io.bids.Bids.get_all_participants_sessions` lists every
``(participant, session)`` pair in the directory, while
:py:meth:`~clinicadl.io.bids.Bids.get_path` and
:py:meth:`~clinicadl.io.bids.Bids.has_file_type` locate a specific file.

1.2.1.2. Describing the files to load: ``BidsFileType``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :py:class:`~clinicadl.io.bids.BidsFileType` defines the :term:`NIfTI` files you are interested
in. It is expressed in the vocabulary of the BIDS specification: a
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your data has been preprocessed with :clinica:`Clinica <>`, you do not have to write
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

1.2.1.3. The ``BidsDataset``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :py:class:`~clinicadl.data.datasets.BidsDataset` is the object you will use most.
It reads a BIDS directory, loads the files described by a ``BidsFileType``, and returns
one :py:class:`~clinicadl.data.structures.Sample` per element of the data.

Consider a dataset whose metadata are stored in a TSV file:

.. code-block:: text

    bids_directory
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
file. This standalone example creates two tiny synthetic images and their metadata
in a temporary directory, so no scan download is needed. These constant-valued
images demonstrate loading only; they are not brain scans or training data.

.. code-block:: python

    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import nibabel as nib
    import numpy as np
    import pandas as pd

    from clinicadl.data.datasets import BidsDataset
    from clinicadl.io.bids import BidsFileType

    with TemporaryDirectory() as tmp_dir:
        bids_directory = Path(tmp_dir)
        description = {
            "Name": "Toy image-loading example",
            "BIDSVersion": "1.10.0",
            "DatasetType": "raw",
        }
        (bids_directory / "dataset_description.json").write_text(
            json.dumps(description), encoding="utf-8"
        )
        metadata = pd.DataFrame(
            {
                "participant_id": ["sub-001", "sub-002"],
                "session_id": ["ses-M000", "ses-M000"],
                "age": [55.0, 62.0],
                "diagnosis": ["control", "patient"],
            }
        )
        metadata.to_csv(bids_directory / "metadata.tsv", sep="\t", index=False)
        for participant in metadata["participant_id"]:
            anat = bids_directory / participant / "ses-M000" / "anat"
            anat.mkdir(parents=True)
            image = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4))
            nib.save(image, anat / f"{participant}_ses-M000_T1w.nii.gz")

        dataset = BidsDataset(
            bids=bids_directory,
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
            data=bids_directory / "metadata.tsv",
        )
        sample = dataset[0]
        print(len(dataset))
        print((sample.participant_id, sample.session_id))
        print(sample.spatial_shape)

This prints:

.. code-block:: text

    2
    ('sub-001', 'ses-M000')
    (8, 8, 8)

Keep dataset access inside the ``with`` block: the temporary files are removed
when it ends. The following snippets illustrate options for your own existing
dataset.

A few arguments shape what a ``BidsDataset`` contains and will output:

``data``
    A :py:class:`pandas.DataFrame` (or a path to a TSV file) listing the
    ``(participant_id, session_id)`` pairs to keep, plus any extra columns. If
    omitted, **all** the pairs that have the requested ``file_type`` are used.

``columns``
    The columns of ``data`` to carry into each ``Sample``. You can pass a list of
    column names, or a dictionary mapping a column name to a function applied to the
    whole column, for instance to encode string labels as integers:

    .. code-block:: python

        import pandas as pd

        def encode_diagnosis(column: pd.Series) -> pd.Series:
            return column.map({"control": 0, "patient": 1})

        dataset = BidsDataset(
            bids="bids_directory",
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
            data="bids_directory/metadata.tsv",
            columns={"age": None, "diagnosis": encode_diagnosis},
        )

    .. code-block:: python

        >>> dataset[0]["diagnosis"]
        0

``masks``
    Masks to load alongside each image, passed as a dictionary. The keys become the
    mask names in the ``Sample`` and the values describe where to find each mask, a
    single shared :term:`NIfTI` file or a :py:class:`~clinicadl.io.bids.BidsFileType` (for a
    participant- and session-specific mask in the same BIDS).

    .. code-block:: text

        bids_directory
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
            bids="bids_directory",
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
            masks={
                "brain": (                                                                   # subject- and session-specific mask that is in another BIDS
                    "bids_directory/derivatives/masks",
                    BidsFileType(
                        data_type="anat", suffix="mask", with_entities={"label": "brain"}
                    ),
                ),
                "mni": "bids_directory/derivatives/registration/space-MNI152NLin2009cSym_mask.nii.gz",  # same mask for all (subject, session)
            },
        )

``transforms``
    A :py:class:`~clinicadl.transforms.TransformsHandler` describing the transforms
    to apply and whether you work on entire images, patches or slices, as we shall see
    in the :doc:`next section <transforms>`:

    .. code-block:: python

        from clinicadl.transforms import TransformsHandler, extraction

        dataset = BidsDataset(
            bids="bids_directory",
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
            data="bids_directory/metadata.tsv",
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

----

The next section presents some advanced tips for neuroimaging data manipulation, like speeding
up data loading, joining multiple datasets (e.g. coming from different cohorts), or reading
non-BIDS-compliant datasets.
