# `train` - Train neural networks using CNN models

* `image`: uses the full 3D MRIs to train a network.
* `patch`: uses 3D patches (from specific patch size) extracted from the 3D image.
* `roi`: extract a specific 3D region from the MRI.
* `slice`: uses 2D slices to train a CNN.

For each mode, different options are presented, in order to control different
parameters used during the training phase.

<details>
<summary>
E.g., this is the list of options available when training a CNN network using
3D patches:
</summary>

```{.sourceCode .bash}
usage: clinicadl train patch cnn [-h] [-cpu] [-np NPROC]
                                 [--batch_size BATCH_SIZE]
                                 [--diagnoses {AD,CN,MCI,sMCI,pMCI} [{AD,CN,MCI,sMCI,pMCI} ...]]
                                 [--baseline] [--n_splits N_SPLITS]
                                 [--split SPLIT [SPLIT ...]] [--epochs EPOCHS]
                                 [--learning_rate LEARNING_RATE]
                                 [--weight_decay WEIGHT_DECAY]
                                 [--dropout DROPOUT] [--patience PATIENCE]
                                 [--tolerance TOLERANCE] [-ps PATCH_SIZE]
                                 [-ss STRIDE_SIZE] [--use_extracted_patches]
                                 [--transfer_learning_path TRANSFER_LEARNING_PATH]
                                 [--transfer_learning_autoencoder]
                                 [--transfer_learning_selection {best_loss,best_acc}]
                                 [--selection_threshold SELECTION_THRESHOLD]
                                 caps_dir {t1-linear,t1-extensive} tsv_path
                                 output_dir network

optional arguments:
  -h, --help            show this help message and exit

Positional arguments:
  caps_dir              Data using CAPS structure.
  {t1-linear,t1-extensive}
                        Defines the type of preprocessing of CAPS data.
  tsv_path              TSV path with subjects/sessions to process.
  output_dir            Folder containing results of the training.
  network               CNN Model to be used during the training.

Computational resources:
  -cpu, --use_cpu       Uses CPU instead of GPU.
  -np NPROC, --nproc NPROC
                        Number of cores used during the training.
  --batch_size BATCH_SIZE
                        Batch size for training. (default=2)

Data management:
  --diagnoses {AD,CN,MCI,sMCI,pMCI} [{AD,CN,MCI,sMCI,pMCI} ...], -d {AD,CN,MCI,sMCI,pMCI} [{AD,CN,MCI,sMCI,pMCI} ...]
                        Diagnoses that will be selected for training.
  --baseline            if True only the baseline is used.

Cross-validation arguments:
  --n_splits N_SPLITS   If a value is given will load data of a k-fold CV.
  --split SPLIT [SPLIT ...]
                        Train the list of given folds. By default train all
                        folds.

Optimization parameters:
  --epochs EPOCHS       Epochs through the data. (default=20)
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate of the optimization. (default=0.01)
  --weight_decay WEIGHT_DECAY, -wd WEIGHT_DECAY
                        Weight decay value used in optimization.
                        (default=1e-4)
  --dropout DROPOUT     rate of dropout that will be applied to dropout
                        layers.
  --patience PATIENCE   Waiting time for early stopping.
  --tolerance TOLERANCE
                        Tolerance value for the early stopping.

Patch-level parameters:
  -ps PATCH_SIZE, --patch_size PATCH_SIZE
                        Patch size
  -ss STRIDE_SIZE, --stride_size STRIDE_SIZE
                        Stride size
  --use_extracted_patches
                        If True the outputs of extract preprocessing are used,
                        else the whole MRI is loaded.

Transfer learning:
  --transfer_learning_path TRANSFER_LEARNING_PATH
                        If an existing path is given, a pretrained model is
                        used.
  --transfer_learning_autoencoder
                        If specified, do transfer learning using an
                        autoencoder else will look for a CNN model.
  --transfer_learning_selection {best_loss,best_acc}
                        If transfer_learning from CNN, chooses which best
                        transfer model is selected.

Patch-level CNN parameters:
  --selection_threshold SELECTION_THRESHOLD
                        Threshold on the balanced accuracies to compute the
                        subject-level performance. Patches are selected if
                        their balanced accuracy > threshold. Default
                        corresponds to no selection.

```
</details>
