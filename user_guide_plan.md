# ClinicaDL User Guide — Working Plan & Handoff

> Scratch/handoff doc for writing the ClinicaDL **User Guide**. Read this first to
> restart the task from where we left off. (Companion to `user_guide_toc.md`, which
> holds the author-approved table of contents and per-section object lists.)

## 1. The problem / goal

ClinicaDL is a Python library (deep learning for neuroimaging, ARAMIS Lab). The
**API Reference** already exists (Sphinx autosummary, rendered to
`docs/_build/html`). The task is to **write the associated User Guide** as new
`.rst` source files under `docs/user_guide/`.

- Repo root: `/Users/thibault.devarax/Desktop/code/clinicadl`
- Library source: `clinicadl/` (~36k LOC, 13 documented modules)
- Docs: `docs/` (Sphinx). Source `.rst` live here; HTML build in `docs/_build/html`
  (generated — never edit).
- Tests: `tests/` (~25k LOC) — usable for real usage patterns.
- Branch: `doc`. Git user: thibaultdvx. Today's date in session: 2026-06-01.

## 2. Requirements & tips the author gave (HARD RULES)

1. **Only mention objects that are in the API Reference.** The API Reference is the
   allowed vocabulary. Full inventory in §5 below. Cross-reference with `:class:` /
   `:func:` / `:mod:` roles into the existing `generated/` pages.
2. **No configuration classes (`*Config`) before section 3.1.** In Chapters 1–2 use
   only raw objects: raw `clinicadl.networks.nn` networks, **PyTorch** losses &
   optimizers, raw transform callables (any function taking & returning a
   `DataPoint`, e.g. TorchIO transforms). The `*Config` pattern is *introduced* in
   §3.1 for the first time.
3. **Add runnable code examples in every section.** Where real data is missing,
   write realistic snippets describing the data as if it existed (e.g. a described
   BIDS tree on disk). **Bundled sample objects exist and SHOULD be used** for
   self-contained runnable examples:
   `clinicadl.data.structures.examples.Colin27DataPoint`, `Colin27Sample`,
   `Colin27Sample2D` (wrap TorchIO's Colin27; need no external data).
4. **Two-level file nesting**: each numbered chapter = a folder with its own
   `index.rst`; each subsection (1.1, 1.2, …) = its own `.rst` file. Third-level
   items (1.2.1, 1.2.2) are headings *within* the subsection file (not separate
   files).
5. Style: match existing docs — Sphinx `furo` theme + `sphinx_design` (grids,
   cards, dropdowns), NumPy docstrings, `::` literal blocks. Extensions incl.
   autosummary, napoleon, autodoc_pydantic, sphinx_copybutton, intersphinx.
   Intersphinx is configured for torch, torchio, monai, pandas, pydantic, numpy,
   nibabel, etc. — so referencing external objects (TorchIO/MONAI/PyTorch) is fine
   and links resolve.
6. Reading the source via **Bash is fine** (cat/grep). Checkpoint: **write
   index + quickstart + all of Chapter 1, then STOP for author review** before
   Chapters 2–4.
7. Naming correction: it is `clinicadl.split.KFold` (the TOC draft said
   `KFoldSplit` — use `KFold`).
8. Do NOT use the `examples/` folder — it is outdated.

## 3. Approved User Guide file structure

```
docs/user_guide/
  index.rst                  User Guide landing (sphinx_design card grid + toctree)
  quickstart/
    index.rst                Quickstart: What is ClinicaDL / Prerequisite / 10 Minutes
  data/                       1. Manipulating neuroimaging data
    index.rst
    structures.rst           1.1 Data structures (DataPoint, Sample, Sample2D)
    bids.rst                 1.2 Reading BIDS datasets (+1.2.1 to_tensors, +1.2.2 joining)
    transforms.rst           1.3 Transforming data (+1.3.1 patches/slices, +1.3.2 pre/aug/post)
    splitting.rst            1.4 Splitting data (+1.4.1 making, +1.4.2 reading)
    dataloader.rst           1.5 Batching data for training
  workflow/                  2. Building a deep learning workflow
    index.rst
    model.rst                2.1 Defining a model (+2.1.1 neural networks)
    training.rst             2.2 Training (+2.2.1 resuming)
    evaluating.rst           2.3 Evaluating
    callbacks.rst            2.4 Callbacks
  reproducibility/           3. Experiment management and reproducibility
    index.rst
    config.rst               3.1 Configuration classes
    maps.rst                 3.2 MAPS
  customising/
    index.rst                4. Customising your ClinicaDL experiment
```

Also: the User Guide must be wired into the site. `docs/index.rst` already links to
`user_guide/index` and `docs/api/index.rst` references `../user_guide/index`. Each
chapter `index.rst` carries the chapter intro + a `toctree` to its subsection files;
`user_guide/index.rst` has a `toctree` to the chapter `index` files. Currently
`docs/user_guide/index.rst` is just a stub title; `docs/getting_started.rst` and
`docs/whats_new.rst` are near-empty stubs too.

## 4. Per-section content map (from user_guide_toc.md)

**Quickstart** — What is ClinicaDL (motivations: accessible high-level entry to DL in
neuroimaging, but flexible Python API; focus on experiment management &
reproducibility for benchmarking; integrates MONAI/TorchIO + BIDS/Clinica).
Prerequisites: data in BIDS, PyTorch, look at MONAI/TorchIO. "10 Minutes to
ClinicaDL" = condensed end-to-end tour summarizing the guide.

**1.1 Data structures** — `data.structures.DataPoint` (image + mask + metadata in one
object), `Sample` & `Sample2D` (children of DataPoint, the output of a Dataset →
transition to next part).

**1.2 Reading BIDS datasets** — `io.bids.Bids`, `io.bids.BidsFileType`,
`data.datasets.BidsDataset`. 1.2.1 Converting NIfTI→tensors to speed up loading:
`BidsDataset.to_tensors` and `data.datasets.TensorDataset`. 1.2.2 Joining datasets:
`ConcatDataset`, `PairedDataset`, `UnpairedDataset`.

**1.3 Transforming data** — `transforms.TransformsHandler`. 1.3.1 Patches & slices:
`transforms.extraction` (`Image`/`Patch`/`Slice`). 1.3.2 Pre/aug/post: ClinicaDL
works with any callable taking & returning a `DataPoint` (use raw transforms here —
NO config classes yet).

**1.4 Splitting data** — importance of longitudinal split in neuroimaging (avoid data
leakage). 1.4.1 Making: `split.make_split`, `split.make_kfold`. 1.4.2 Reading:
`split.SingleSplit`, `split.KFold`, `split.Split`.

**1.5 Batching** — `data.dataloader` (`DataLoader`, `Batch`, `CollateFn`).

**2.1 Defining a model** — `clinicadl.models` (`Model`, `SupervisedModel`,
`ReconstructionModel`); losses & optimizers come from PyTorch; mention `clinicadl.infer`
for evaluation. 2.1.1 Neural networks: `clinicadl.networks.nn` (many available).

**2.2 Training** — `clinicadl.train` (`Trainer`). Do NOT mention `Trainer.validate`
/ `Trainer.test` here. Mention metrics briefly, defer to 2.3. 2.2.1 Resuming:
`Trainer.resume`.

**2.3 Evaluating** — `metrics.Metric`, `metrics.MetricsHandler` (briefly); quick
reference to `clinicadl.infer` for customizing inference/evaluation; here is where
`Trainer.validate` and `Trainer.test` are introduced.

**2.4 Callbacks** — `clinicadl.callbacks`.

**3.1 Configuration classes** — a dataclass associated to an object holding only its
parameters (almost no logic). Easy to save/read → reproducibility. Access the
underlying object via `get_object`. Advise working with config classes over raw
objects for best reproducibility. The list of available config classes is not
exhaustive (mostly for MONAI/TorchIO objects) but will grow. **This is where config
classes are first introduced.**

**3.2 MAPS** — `io.maps.Maps`. Contains all outputs of training/evaluation (experiment
management: all results in one shareable folder) and all hyperparameters used
(reproducibility; caveat: objects from outside ClinicaDL like TorchIO/MONAI work but
aren't 100% reproducible).

**4. Customising** — mention callbacks briefly (ref 2.4); OOP: many ClinicaDL objects
can be subclassed to modify behaviour (e.g. `Model`).

## 5. API Reference inventory (the ALLOWED vocabulary)

- **callbacks**: `Callback`, `CallbacksHandler`, `EarlyStoppingCallback`,
  `LoggerCallback`, `LRSchedulerCallback`, `ModelCheckpointCallback`,
  `MonitorCallback`, `TrainingCheckpointCallback`
- **data.datasets**: `Dataset`, `BidsDataset`, `TensorDataset`, `ConcatDataset`,
  `PairedDataset`, `UnpairedDataset`
- **data.dataloader**: `Batch`, `DataLoader`; collate: `CollateFn`, `ToBatchCollate`,
  `ToBatchesCollate`, `MergeBatchesCollate`
- **data.structures**: `DataPoint`, `Sample`, `Sample2D`; examples:
  `Colin27DataPoint`, `Colin27Sample`, `Colin27Sample2D`
- **data.utils**: `remove_tensors`
- **io.bids**: `Bids`, `BidsFileType`, `T1Linear`, `FlairLinear`, `PetLinear`, `DwiDti`
- **io.maps**: `Maps`
- **infer**: `Inferer`, `SimpleInferer`, `SlicesToImageInferer`, `PatchesToImageInferer`
- **losses.config**: BCE/BCEWithLogits/CrossEntropy/NLL/MultiMargin (classif/seg);
  MSE/L1/SmoothL1/Huber/KLDiv (regr/recon) `*Config`
- **metrics**: `Metric`, `MetricsHandler`; **metrics.config**: LossMetric;
  ConfusionMatrix/ROCAUC/AveragePrecision (classif); MSE/MAE/RMSE (regr);
  PSNR/SSIM/MultiScaleSSIM (recon); MeanIoU/Dice/GeneralizedDiceScore/SurfaceDice/
  SurfaceDistance/HausdorffDistance (seg) `*Config`
- **models**: `Model`, `SupervisedModel`, `ReconstructionModel`
- **networks.nn**: builders `MLP`, `ConvEncoder`, `ConvDecoder`, `CNN`, `Generator`,
  `AutoEncoder`, `VAE`; common `UNet`, `AttentionUNet`, `DenseNet`, `ResNet`,
  `SEResNet`, `ViT`; literature `DenseNet121/161/169/201`, `ResNet18/34/50/101/152`,
  `SEResNet50/101/152`, `ViTB16/B32/L16/L32`. **networks.config**: matching `*Config`
- **optim**: `OptimizationConfig`; optimizers.config: Adadelta/Adagrad/Adam/RMSprop/
  SGD `Config`; lr_schedulers.config: ConstantLR/ExponentialLR/LinearLR/StepLR/
  MultiStepLR/PolynomialLR/ReduceLROnPlateau/OneCycleLR `Config`
- **split**: `make_split`, `make_kfold`, `SingleSplit`, `KFold`, `Split`
- **transforms**: `TransformsHandler`, `MergeFields`, `Format`; extraction: `Image`,
  `Patch`, `Slice`; **transforms.config**: preprocessing (RescaleIntensity,
  ZNormalization, Mask, Clamp, CropOrPad, ToCanonical, Resize, Resample,
  EnsureShapeMultiple, Crop, Pad, RemapLabels, OneHot, MergeFields), augmentation
  (OneOf, RandomMotion/Ghosting/Spike/BiasField/Blur/Noise/Swap/Gamma/Flip/Affine/
  ElasticDeformation/Anisotropy), post-processing (Activations, AsDiscrete,
  KeepLargestConnectedComponent, DistanceTransformEDT, RemoveSmallObjects,
  LabelFilter, FillHoles, SobelGradients, Format) `*Config`
- **train**: `Trainer`, `TrainerState`, `ComputationalConfig`
- **utils**: `seed.seed_everything`, `seed.seed_everything_context`

## 6. Strategy to execute

- Read source **lazily, per subsection** via Bash (cat/grep): just that module's
  public classes (signature, defaults, docstring) + one real usage in `tests/`,
  then write the `.rst`. Docstrings are rich (many contain runnable doctest
  examples using Colin27) — reuse/adapt them. Don't ingest all 36k LOC.
- Build the folder tree, then write **index + quickstart + Chapter 1**, STOP for
  review (per requirement #6).
- Reconcile any TOC-vs-source naming against the source (e.g. KFold).

## 7. Progress so far (source already read & understood)

- Mapped full API Reference (all `docs/api/**/*.rst`) → §5 inventory.
- Read scaffolding: `docs/index.rst`, `docs/conf.py`, `installation.rst`,
  `glossary.rst`, `getting_started.rst` (stub), `user_guide/index.rst` (stub).
- Read source: `data/structures/{datapoint,sample,examples}.py` (DataPoint/Sample/
  Sample2D + Colin27 examples) ✔; `data/datasets/{base,bids,tensor}.py`
  (Dataset, BidsDataset incl. `to_tensors`, TensorDataset) ✔; `io/bids/reader.py`
  (Bids) + `io/bids/file_type/{base,clinica}.py` (BidsFileType, T1Linear/FlairLinear/
  PetLinear/DwiDti) ✔; `transforms/__init__.py`, `transforms/handlers/transforms.py`
  (TransformsHandler: extraction + image_transforms + sample_transforms +
  augmentations), `transforms/extraction/__init__.py`, `transforms/types.py`
  (Transform = Callable[[DataPoint], DataPoint]), `transforms/homemade.py` (Format) ✔.

### Still to read before writing Chapter 1
- `transforms/extraction/{image,patch,slice}.py` (Image/Patch/Slice signatures:
  patch_size/stride, slice_direction, etc.) — **next step**.
- `data/datasets/{concat,paired,unpaired}.py` (joining datasets, §1.2.2).
- `split/` (`make_splits/`, `splitter/`, `split.py`) for §1.4.
- `data/dataloader/{loader,batch}.py` + `collate/` for §1.5.

### Then (Chapters 2–4, read when reached)
- `models/` (Model/SupervisedModel/ReconstructionModel) + `networks/nn/`.
- `train/` (Trainer; note: hide validate/test in 2.2, show in 2.3) + `callbacks/`.
- `metrics/` + `infer/`.
- config-class base (`utils/config.py`, `get_object`) for §3.1; `io/maps/` for §3.2.
- `models/` subclassing + `utils/seed.py` for ch.4 / reproducibility refs.

## 8. Key technical facts learned (use when writing)

- `DataPoint(image, participant, session, **kwargs)` — subclass of
  `torchio.Subject` (a dict). Attrs: `image` (tio.ScalarImage), `participant`,
  `session`; helpers: `add_image`/`add_mask`, `get_images_dict`/`get_masks_dict`/
  `get_non_images_dict`, `get_image_tensor`, `shape`/`spatial_shape`/`spacing`/
  `affine`, `plot`. `Colin27DataPoint()` → keys (head, image, participant, session),
  shape (1,181,217,181).
- `Sample(DataPoint)` = output of a `Dataset`; adds `file_type`, `image_path`,
  `sample_type` (image/patch/slice), `sample_position`. `Sample2D` = a slice; adds
  `slice_direction` (0 sagittal/1 coronal/2 axial), `squeeze`.
- `Bids(path)` reads a BIDS / derivative / CAPS (via `dataset_description.json`
  `DatasetType`). Methods: `get_path`, `has_file_type`, `build_path`,
  `get_participants_sessions_with`, `get_all_participants_sessions`; props
  `participants_dir`, `tensors_dir`.
- `BidsFileType(suffix, data_type=None, extension=.nii.*, with_entities=None,
  without_entities=None, description=None)`; `.match(path, participant, session)`.
  Concrete: `T1Linear(use_uncropped_image=False)`, `FlairLinear(...)`,
  `PetLinear(tracer, suvr_reference_region, reconstruction=None, use_uncropped_image=False)`,
  `DwiDti(measure, space)`.
- `BidsDataset(bids, file_type, data=None, transforms=TransformsHandler(),
  columns=None, masks=None)`. `data` = DataFrame/TSV of (participant_id, session_id)
  [+ extra cols]; `columns` = list or {col: fn(Series)->Series}; `masks` = dict
  (path | BidsFileType | (Bids, BidsFileType)). Length = #images × #samples-per-image.
  `.df`, `.subset(...)`, `.to_tensors(conversion_name, save_transforms, n_proc, ...)`
  → `TensorDataset`. `TensorDataset(description_json, data=None, transforms=...,
  columns=None, to_load=None)` reads `.pt` (much faster loading).
- `TransformsHandler(extraction=Image(), image_transforms=[], sample_transforms=[],
  augmentations=[])`. 4 transform kinds: extraction (image/patch/slice);
  image_transforms (whole image, BEFORE extraction — normalization here);
  sample_transforms (AFTER extraction — resize to fit network); augmentations
  (only during training). Accepts raw callables (Transform = Callable[[DataPoint],
  DataPoint]) OR config classes. In Chapter 1 use raw callables only (e.g. TorchIO
  transforms, custom functions).
- A `Dataset` (`data.datasets.Dataset`) is abstract, subclass of
  `torch.utils.data.Dataset`; has `.df`, `.train()`/`.eval()` modes, `.subset(...)`,
  `__getitem__` → `Sample`.
