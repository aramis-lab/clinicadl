# `train slice` - Train classification CNN using 2D slices

This option allows training a network on 2D slices. For more information on slices please refer to [tensor extraction](../Extract.md).
There is no network type choice for `slice` as the only network type is the single-CNN.

!!! note
    Options that are common to all pipelines can be found in the introduction of [`clinicadl train`](./Introduction.md#running-the-pipeline)

The objective of this unique CNN is to learn to predict labels associated to images.
The set of images used corresponds to all the possible slice locations in MR volumes.
Slices at the beginning or at the end of the volume may be excluded using the `discarded_slices` argument.

The output of the CNN is a vector of size equals to the number of classes in this dataset.
This vector can be preprocessed by the [softmax function](https://pytorch.org/docs/master/generated/torch.nn.Softmax.html) 
to produce a probability for each class. During training, the CNN is optimized according to the cross-entropy loss, 
which becomes null for a subset of images if the CNN outputs 100% probability for the true class of each image of the subset.

The options specific to this pipeline are the following:

- `--slice_direction` (int) axis along which the MR volume is sliced. Default: `0`.
    - 0 corresponds to sagittal plane,
    - 1 corresponds to coronal plane,
    - 2 corresponds to axial plane.
- `--discarded_slices` (list of int) number of slices discarded from respectively the beginning and the end of the MRI volume. 
If only one argument is given, it will be used for both sides. Default: `20`.
- `--use_extracted_slices` (bool) if this flag is given, the outputs of `clinicadl extract` are used.
Else the whole 3D MR volumes are loaded and slices are extracted on-the-fly.
- `--selection_threshold` (float) threshold on the balanced accuracies to compute the image-level performance. 
Slices are selected if their balanced accuracy > threshold. Default corresponds to no selection.

!!! note
    The selection of a best model is only performed at the end of an epoch 
    (a model cannot be selected based on internal evaluations in an epoch).

The complete output file system is the following:

<pre>
results
├── commandline.json
├── environment.txt
└── fold-0
    ├── cnn_classification
    │   ├── best_balanced_accuracy
    │   │   ├── train_image_level_metrics.tsv
    │   │   ├── train_image_level_prediction.tsv
    │   │   ├── train_slice_level_metrics.tsv
    │   │   ├── train_slice_level_prediction.tsv
    │   │   ├── validation_image_level_metrics.tsv
    │   │   ├── validation_image_level_prediction.tsv
    │   │   ├── validation_slice_level_metrics.tsv
    │   │   └── validation_slice_level_prediction.tsv
    │   └── best_loss
    │       └── ...
    ├── models
    │   ├── best_balanced_accuracy
    │   │   └── model_best.pth.tar
    │   └── best_loss
    │       └── model_best.pth.tar
    └── tensorboard_logs
         ├── train
         │    └── events.out.tfevents.XXXX
         └── validation
              └── events.out.tfevents.XXXX
</pre>

!!! note
    The performances are obtained at two different levels: slice-level and image-level. 
    Slice-level performance corresponds to an evaluation in which all slices are considered to be independent. 
    However it is not the case, and what is more interesting is the evaluation on the image-level, 
    for which the predictions of slice-level were assembled.