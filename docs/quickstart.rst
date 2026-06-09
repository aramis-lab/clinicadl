.. _user_guide_quickstart:

Quickstart
==========

What is ClinicaDL?
------------------

**ClinicaDL** is an open-source Python library for deep learning in neuroimaging.
It is built around four ideas:

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: :fas:`door-open` Accessibility

        ClinicaDL offers a **high-level** entry point to deep learning in
        neuroimaging. Common operations such as reading a dataset, extracting patches or
        slices, building a longitudinal split, and training a model, are available out
        of the box, so that you can easily build a functional deep learning workflow
        without confronting the full complexity of native PyTorch.

    .. grid-item-card:: :fas:`sliders` Flexibility

        ClinicaDL is a **Python API**, not a black box. It provides a broad range of ready-to-use
        Python objects, while also allowing advanced users to leverage object-oriented programming
        to design new objects to tailor their workflows.

    .. grid-item-card:: :fas:`box-archive` Reproducibility

        ClinicaDL puts a strong emphasis on **experiment management** and
        **reproducibility**. All the outputs and hyperparameters of an experiment
        are gathered in a single folder, which makes ClinicaDL well suited to
        **benchmarking**.

    .. grid-item-card:: :fas:`puzzle-piece` Integration

        ClinicaDL builds on the existing **ecosystem**. It is developed on top of :torchio:`TorchIO <>` and
        :monai:`MONAI <>`, and interoperates seamlessly with these libraries. It also relies on the neuroimaging
        community standard :term:`BIDS` and is compatible with :clinica:`Clinica <>`, enabling end-to-end pipelines
        from pre-processing to model training and evaluation.

Prerequisites
-------------

To work with ClinicaDL, you should:

- **Organise your data following BIDS standard**. The Brain Imaging Data Structure
  (`BIDS <https://bids.neuroimaging.io/index.html>`_) is a community standard that
  describes how neuroimaging files should be named and arranged on disk. This standardized
  layout makes it easy for tools to automatically locate and query data within a BIDS dataset.
  ClinicaDL also supports :term:`BIDS derivatives <BIDS derivative>` and the :term:`CAPS` directories
  produced by :clinica:`Clinica <>`.
- **Be familiar with the basics of** :torch:`PyTorch <>`. ClinicaDL is built on top of
  PyTorch (datasets, dataloaders, modules, losses and optimizers all wrap standard
  PyTorch tools), and you are free to use raw PyTorch objects throughout.
  Although ClinicaDL saves you from writing your pipeline from scratch, it is still a good idea
  to understand the typical steps of a PyTorch deep learning workflow.
- Optionally, **have a look at** :torchio:`TorchIO <>` **and** :monai:`MONAI <>`. ClinicaDL builds
  upon these libraries and provides seamless access to their objects.

For installation instructions, see the :doc:`installation page <../../installation>`.

10 minutes to ClinicaDL
-----------------------

The rest of this page is a condensed, end-to-end tour of a typical ClinicaDL
experiment: reading and preprocessing data, splitting it, training a model,
evaluating it, and managing the results.

Manipulating neuroimaging data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read your images straight from a :term:`BIDS` dataset, selecting exactly the files you
need. Each image is loaded together with its metadata and any associated masks, keeping everything
together in a single object that moves consistently through the rest of the pipeline

.. code-block:: python

    from clinicadl.data.datasets import BidsDataset
    from clinicadl.io.bids import BidsFileType

    dataset = BidsDataset(
        bids="bids_directory",                  # path to your BIDS directory
        file_type=BidsFileType(
            data_type="anat", suffix="T1w"
        ),                                      # type of data to consider in your BIDS
        data="bids_directory/metadata.tsv",               # (participant, session) pairs + metadata
        columns=["gender", "age"],              # metadata to carry along the images
        masks={
            "csf": BidsFileType(
                data_type="anat", suffix="mask", with_entities={"label": "csf"}
            )
        },                                      # masks to carry along the image
    )

.. code-block:: bash
    :caption: bids_directory/metadata.tsv

    participant_id  session_id   age   sex
    sub-001         ses-M000     55.0  M
    sub-002         ses-M000     62.0  F
    sub-002         ses-M024     64.0  F
    sub-003         ses-M000     67.0  F
    ...

Patches and slices extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to extract 3D patches or 2D slices from the 3D images.

.. code-block:: python

    import torchio as tio
    from clinicadl.transforms import TransformsHandler, extraction

    # a dataset that loads images and extracts 64x64x64 patches
    dataset = BidsDataset(
        bids="bids_directory",
        file_type=BidsFileType(
            data_type="anat", suffix="T1w"
        ),
        transforms=TransformsHandler(
            extraction=extraction.Patch(patch_size=64)
        ),
    )

Transforms
^^^^^^^^^^

Transform your images and augment them on the fly. You can decide to apply your transform
at the image or the sample level (i.e. on patches or slices).

.. code-block:: python

    import torchio as tio

    dataset = BidsDataset(
        bids="bids_directory",
        file_type=BidsFileType(data_type="anat", suffix="T1w"),
        transforms=TransformsHandler(
            image_transforms=[tio.ToCanonical(), tio.ZNormalization()],   # applied to the whole image
            extraction=extraction.Slice(slice_direction=0),               # gets all the slices in the sagittal direction
            sample_transforms=[tio.CropOrPad(target_shape=(1, 64, 64))],  # applied to a slice
            augmentations=[tio.RandomFlip()],                             # applied only during training
        ),
    )

Data splitting
^^^^^^^^^^^^^^

Separate your **participants** into training, validation and test sets. The split is done at the participant-level,
meaning that all the sessions of a participant stay in the same set, avoiding :term:`data leakage`.

.. code-block:: python

    from clinicadl.split import make_split, make_kfold, KFold

    # split participants in train/validation/test sets
    split_dir = make_split("bids_directory/metadata.tsv", n_test=0.2, seed=42)
    kfold_dir = make_kfold(split_dir / "training.tsv", n_splits=5, seed=42)

    # now split any dataset according to this splitting
    splitter = KFold(split_dir)

    # iterate over splits
    for split in splitter.get_splits(dataset):
        split.build_train_loader(batch_size=8, shuffle=True)
        split.build_val_loader(batch_size=8)
        ...

Training a neural network
~~~~~~~~~~~~~~~~~~~~~~~~~

Assemble a model from a neural network, a loss and an optimizer, then train it.
ClinicaDL comes with a **trainer** that handles most of the PyTorch complexity and provides some
acceleration features. The trainer can be customised via **callbacks**, which are objects
that modifies the non-essential logic of the training (e.g., learning rate scheduling, early stopping).

.. code-block:: python

    from monai.losses import DiceLoss
    from clinicadl.callbacks import EarlyStoppingCallback, ModelCheckpointCallback
    from clinicadl.infer import PatchesToImageInferer
    from clinicadl.metrics.config import AveragePrecisionMetricConfig, DiceMetricConfig
    from clinicadl.models import SupervisedModel
    from clinicadl.networks.nn import AttentionUNet
    from clinicadl.optim.optimizers.config import AdamConfig
    from clinicadl.train import Trainer, OptimizationConfig

    segmentation_model = SupervisedModel(
        network=AttentionUNet(spatial_dims=3, in_channels=1, out_channels=1),
        loss=DiceLoss(),
        optimizer=AdamConfig(),
        inferer=PatchesToImageInferer(patch_size=64, overlap=0.25),
        label_key="csf",
    )

    trainer = Trainer(
        maps="maps_directory",
        model=segmentation_model,
        callbacks=[
            EarlyStoppingCallback(metric="loss", patience=10),
            ModelCheckpointCallback(epochs=range(0, 100, 10)),
        ],
        metrics={"dice": DiceMetricConfig(), "ap": AveragePrecisionMetricConfig()},
        optimization=OptimizationConfig(epochs=100),
    )

    trainer.train(
        split=split,
        computational=ComputationalConfig(gpu=True, amp=True, seed=42, deterministic=True),
    )

Note in the above example the convenient ``PatchesToImageInferer``, which splits a 3D image into patches,
passes them through the neural network, and merges the resulting segmentation masks in a single
3D output.

.. seealso::

    See :ref:`Experiment management <exp_management>` for more details on the argument ``maps``.

Model evaluation
~~~~~~~~~~~~~~~~

Evaluate a trained model on held-out data and record the metrics. You point to a
saved checkpoint, provide the data to evaluate on, and the metrics to compute.

.. code-block:: python

    from clinicadl.data.dataloader import DataLoader

    test_dataset = dataset.subset(split_dir / "test_baseline.tsv")
    test_loader = DataLoader(test_dataset)

   # evaluates the final model trained on split 0
    trainer.test(
        model_checkpoint="split-0_final", group_name="test", dataloader=test_loader, metrics=["dice"]
    )

.. _exp_management:

Experiment management
~~~~~~~~~~~~~~~~~~~~~

All outputs, trained weights, metrics, logs and the
configuration used, are gathered into a single folder (a :term:`MAPS`), which fully describes your
experiment. So your results are **easy to share** and your experiments **easy to
reproduce**. You can re-open a past experiment at any time, for example to
resume an interrupted training:

.. code-block:: python

    trainer = Trainer.from_maps("maps_directory")
    trainer.resume(split_idx=0)

Compatibility with community tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At many points, ClinicaDL allows you to provide objects that you have created yourself or that come from external
libraries such as **PyTorch**, **MONAI**, or **TorchIO**. However, since these are external objects, ClinicaDL cannot
guarantee reproducibility when they are used (e.g., :py:meth:`clinicadl.train.Trainer.from_maps` will
not be able to read these objects).

When available, you can instead use the corresponding **configuration classes**, which store the parameters of the raw objects
and can be easily saved, shared, and reloaded by ClinicaDL. For example,

.. code-block:: python

    from torchio import CropOrPad

    transforms = TransformsHandler(image_transforms=[CropOrPad(target_shape=(1, 64, 64))])

is equivalent to:

.. code-block:: python

    from clinicadl.transforms.config import CropOrPadConfig

    transforms = TransformsHandler(image_transforms=[CropOrPadConfig(target_shape=(1, 64, 64))])

Next steps
----------

We have briefly covered the main features of ClinicaDL.
Please refer to our :doc:`user guide <../index>` for more details.

To start using ``ClinicaDL`` on your own, you will find useful help in the
:doc:`../../api/index`.