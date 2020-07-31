# `train patch` - Train deep learning networks using 3D patches

This option allows training a network on 3D patches. For more information on patches please refer to [tensor extraction](../Extract.md).
Three network types can be trained with the `patch` input type:

- `autoencoder`, that trains one autoencoder on all patch locations,
- `cnn`, that trains one CNN on all patch locations,
- `multicnn`, that trains one CNN per patch location.

It is possible to transfer trainable parameters between models. In the following list the weights are transferred from `source task` to `target task`:

- `autoencoder` to `cnn`: the trainable parameters of the convolutional part of the `cnn` 
(convolutions and batch normalization layers) take the values of the trainable parameters of the encoder part of the source autoencoder,
- `cnn` to `cnn`: all the trainable parameters are transferred between the two models.
- `autoencoder` to `multicnn`: the convolutional part of each CNN of the `multicnn` run is initialized
 with the weights of the encoder of the source autoencoder.
- `cnn` to `multicnn`: each CNN of the `multicnn` run is initialized with the weights of the source CNN.
- `multicnn` to `multicnn`: each CNN is initialized with the weights of the corresponding one in the source experiment.

!!! note
    Options that are common to all pipelines can be found in the introduction of [`clinicadl train`](./Introduction.md#running-the-pipeline)

## `train patch autoencoder` - Train autoencoders using 3D patches

The objective of an autoencoder is to learn to reconstruct images given in input while performing a dimension reduction. 

The difference between the input and the output image is given by the mean squared error.
In clinicadl, autoencoders are designed [based on a CNN architecture](./Introduction.md#autoencoders-construction-from-cnn-architectures). 

The options specific to this pipeline are the following: 

- `--patch_size` (int) size of the patches. Default: `50`.
- `--stride_size` (int) length between the centers of successive patches. Default: `50`.
- `--use_extracted_patches` (bool) if this flag is given, the outputs of `clinicadl extract` are used.
Else the whole 3D MR volumes are loaded and patches are extracted on-the-fly.
- `--visualization` (bool) if this flag is given, inputs of the train and
the validation sets and their corresponding reconstructions are written in `autoencoder_reconstruction`.
Inputs are reconstructed based on the model that obtained the best validation loss.

!!! note
    The selection of a best model is only performed at the end of an epoch (a model cannot be selected based on internal evaluations in an epoch).

The complete output file system is the following (the folder `autoencoder_reconstruction` is created only if the 
flag `--visualization` was given):

<pre>
results
├── commandline.json
├── environment.txt
└── fold-0
    ├── autoencoder_reconstruction
    │   ├── train
    │   │   ├── input-0.nii.gz
    │   │   ├── ...
    │   │   ├── input-&lt;N&gt;.nii.gz
    │   │   ├── output-0.nii.gz
    │   │   ├── ...
    │   │   └── output-&lt;N&gt;.nii.gz
    │   └── validation
    │        ├── input-0.nii.gz
    │        ├── ...
    │        ├── input-&lt;N&gt;.nii.gz
    │        ├── output-0.nii.gz
    │        ├── ...
    │        └── output-&lt;N&gt;.nii.gz
    ├── models
    │    └── best_loss
    │        └── model_best.pth.tar
    └── tensorboard_logs
         ├── train
         │    └── events.out.tfevents.XXXX
         └── validation
              └── events.out.tfevents.XXXX
</pre>

`autoencoder_reconstruction` contains the reconstructions of all the patches of the first image of the dataset.
The number of patches `N` depends on the `patch_size` and the `stride_size`.

## `train patch cnn` - Train classification CNN using all 3D patches

The objective of this unique CNN is to learn to predict labels associated to images.
The set of images used corresponds to all the possible patch locations in MR volumes.

The output of the CNN is a vector of size equals to the number of classes in this dataset.
This vector can be preprocessed by the [softmax function](https://pytorch.org/docs/master/generated/torch.nn.Softmax.html) 
to produce a probability for each class. During training, the CNN is optimized according to the cross-entropy loss, 
which becomes null for a subset of images if the CNN outputs 100% probability for the true class of each image of the subset.

The options specific to this pipeline are the following:

- `--patch_size` (int) size of the patches. Default: `50`.
- `--stride_size` (int) length between the centers of successive patches. Default: `50`.
- `--use_extracted_patches` (bool) if this flag is given, the outputs of `clinicadl extract` are used.
Else the whole 3D MR volumes are loaded and patches are extracted on-the-fly.
- `--transfer_learning_path` (str) is the path to a results folder (output of `clinicadl train`). 
The best model of this folder will be used to initialize the network as explained in the [introduction](./Image.md). 
If nothing is given the initialization will be random.
- `--transfer_learning_selection` (str) corresponds to the metric according to which the best model of `transfer_learning_path` will be loaded. 
This argument will only be taken into account if the source network is a CNN. 
Choices are `best_loss` and `bset_balanced_accuracy`.  Default: `best_balanced_accuracy`.
- `--selection_threshold` (float) threshold on the balanced accuracies to compute the image-level performance. 
Patches are selected if their balanced accuracy > threshold. Default corresponds to no selection.

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
    │   │   ├── train_patch_level_metrics.tsv
    │   │   ├── train_patch_level_prediction.tsv
    │   │   ├── validation_image_level_metrics.tsv
    │   │   ├── validation_image_level_prediction.tsv
    │   │   ├── validation_patch_level_metrics.tsv
    │   │   └── validation_patch_level_prediction.tsv
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
    The performances are obtained at two different levels: patch-level and image-level. 
    Patch-level performance corresponds to an evaluation in which all patches are considered to be independent. 
    However it is not the case, and what is more interesting is the evaluation on the image-level, 
    for which the predictions of patch-level were assembled.

## `train patch multicnn` - Train one classification CNN per patche location

Contrary to the preceding pipeline in which all patch locations were used as input of one unique CNN, with this option
one CNN is trained per patch location. Then the result

The output of each CNN is a vector of size equals to the number of classes in this dataset.
This vector can be preprocessed by the [softmax function](https://pytorch.org/docs/master/generated/torch.nn.Softmax.html) 
to produce a probability for each class. During training, the CNN is optimized according to the cross-entropy loss, 
which becomes null for a subset of images if the CNN outputs 100% probability for the true class of each image of the subset.

The options specific to this pipeline are the following:

- `--transfer_learning_path` (str) is the path to a results folder (output of `clinicadl train`). 
The best model of this folder will be used to initialize the network as explained in the [introduction](./Image.md). 
If nothing is given the initialization will be random.
- `--transfer_learning_selection` (str) corresponds to the metric according to which the best model of `transfer_learning_path` will be loaded. 
This argument will only be taken into account if the source network is a CNN. Choices are `best_loss` and `bset_balanced_accuracy`.  
Default: `best_balanced_accuracy`.

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
    │   │   ├── train_patch_level_metrics.tsv
    │   │   ├── train_patch_level_prediction.tsv
    │   │   ├── validation_image_level_metrics.tsv
    │   │   ├── validation_image_level_prediction.tsv
    │   │   ├── validation_patch_level_metrics.tsv
    │   │   └── validation_patch_level_prediction.tsv
    │   └── best_loss
    │       └── ...
    ├── models
    │   ├── cnn-0
    │   │   ├── best_balanced_accuracy
    │   │   │   └── model_best.pth.tar
    │   │   └── best_loss
    │   │       └── model_best.pth.tar
    │   ├── ...
    │   └── cnn-&lt;N&gt;
    │       ├── best_balanced_accuracy
    │       │   └── model_best.pth.tar
    │       └── best_loss
    │           └── model_best.pth.tar    
    └── tensorboard_logs
        ├── cnn-0
        │   ├── train
        │   │   └── events.out.tfevents.XXXX
        │   └── validation
        │       └── events.out.tfevents.XXXX
        ├── ...
        └── cnn-&lt;N&gt;
            ├── train
            │   └── events.out.tfevents.XXXX
            └── validation
                └── events.out.tfevents.XXXX
</pre>

`models` and `tensorboard_logs` contain one output per CNN trained. 
The number of networks (equals to the number of patches) `N` depends on the `patch_size` and the `stride_size`.

!!! note
    The performances are obtained at two different levels: patch-level and image-level. 
    Patch-level performance corresponds to the concatenation of the performances of all CNNs. 
    The evaluation on the image-level is obtained by assembling the predictions of all the CNNs.
