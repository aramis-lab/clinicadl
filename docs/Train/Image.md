# `train image` - Train deep learning networks using whole 3D images

This option allows training a network on the whole 3D images (as opposed to 3D patches and 2D slices).
Two network types can be trained with the `image` mode:

- `autoencoder`, that learns to reconstruct the input images,
- `cnn`, that learns to differentiate two labels assigned to input images. 

It is possible to transfer trainable parameters between models. In the following list the weights are transferred from `source task` to `target task`:

- `autoencoder` to `cnn`: the trainable parameters of the convolutional part of the `cnn` 
(convolutions and batch normalization layers) take the values of the trainable parameters of the encoder part of the source autoencoder,
- `cnn` to `cnn`: all the trainable parameters are transferred between the two models. 

!!! info "Common options"
    Options that are common to all pipelines can be found in the introduction of [`clinicadl train`](./Introduction.md#running-the-pipeline)

## `train image autoencoder` - Train autoencoders using whole 3D images

The objective of an autoencoder is to learn to reconstruct images given in input while performing a dimension reduction. 

The difference between the input and the output image is given by the mean squared error.
In clinicadl, autoencoders are designed [based on a CNN architecture](./Introduction.md#autoencoders-construction-from-cnn-architectures). 

There is one specific option for this pipeline: 

- `--visualization` (bool) if this flag is given, inputs of the train and
the validation sets and their corresponding reconstructions are written in `autoencoder_reconstruction`.
Inputs are reconstructed based on the model that obtained the best validation loss.

??? note "Model selection"
    The selection of a best model is only performed at the end of an epoch 
    (a model cannot be selected based on internal evaluations in an epoch).

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
    │   │   ├── input-1.nii.gz
    │   │   ├── input-2.nii.gz
    │   │   ├── output-0.nii.gz
    │   │   ├── output-1.nii.gz
    │   │   └── output-2.nii.gz
    │   └── validation
    │        ├── input-0.nii.gz
    │        ├── input-1.nii.gz
    │        ├── input-2.nii.gz
    │        ├── output-0.nii.gz
    │        ├── output-1.nii.gz
    │        └── output-2.nii.gz
    ├── models
    │    └── best_loss
    │        └── model_best.pth.tar
    └── tensorboard_logs
         ├── train
         │    └── events.out.tfevents.XXXX
         └── validation
              └── events.out.tfevents.XXXX
</pre>

`autoencoder_reconstruction` contains the reconstructions of the three first participants of the dataset.

## `train image cnn` - Train classification CNN using whole 3D images

The objective of a CNN is to learn to predict labels associated to images. 

The output of a CNN is a vector of size equals to the number of classes in this dataset. 
This vector can be preprocessed by the [softmax function](https://pytorch.org/docs/master/generated/torch.nn.Softmax.html) 
to produce a probability for each class. 
During training, the CNN is optimized according to the cross-entropy loss, which becomes null for a subset of images 
if the CNN outputs 100% probability for the true class of each image of the subset.

The options specific to this pipeline are the following:

- `--transfer_learning_path` (str) is the path to a results folder (output of `clinicadl train`). 
The best model of this folder will be used to initialize the network as explained in the [introduction](./Image.md). 
If nothing is given the initialization will be random.
- `--transfer_learning_selection` (str) corresponds to the metric according to which the best model of `transfer_learning_path` will be loaded. 
This argument will only be taken into account if the source network is a CNN. Choices are `best_loss` and `best_balanced_accuracy`.  
Default: `best_balanced_accuracy`.

??? note "Model selection"
    The selection of a best model is only performed at the end of an epoch 
    (a model cannot be selected based on internal evaluations in an epoch).

The complete output file system is the following:

<pre>
results
├── commandline.json
├── environment.txt
└── fold-0
    ├── cnn_classification
    │   ├── best_balanced_accuracy
    │   │   ├── train_image_level_metrics.tsv
    │   │   ├── train_image_level_prediction.tsv
    │   │   ├── validation_image_level_metrics.tsv
    │   │   └── validation_image_level_prediction.tsv
    │   └── best_loss
    │        ├── train_image_level_metrics.tsv
    │        ├── train_image_level_prediction.tsv
    │        ├── validation_image_level_metrics.tsv
    │        └── validation_image_level_prediction.tsv
    ├── models
    │    └── best_loss
    │        └── model_best.pth.tar
    └── tensorboard_logs
         ├── train
         │    └── events.out.tfevents.XXXX
         └── validation
              └── events.out.tfevents.XXXX
</pre>
