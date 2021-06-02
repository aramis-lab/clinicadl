# Implementation details

Details of implementation corresponding to modules used in the provided architectures, autoencoder construction, 
transfer learning or training details are provided in this section.

## Adaptive padding in pooling layers

Pooling layers reduce the size of their input feature maps. 
There are no learnable parameters in this layer, the kernel outputting the maximum value of the part of the feature map its kernels is covering.

Here is a 2D example of the standard layer of pytorch `nn.MaxPool2d`:

<img src="https://drive.google.com/uc?id=1qh9M9r9mfpZeSD1VjOGQAl8zWqBLmcKz" style="height: 200px;" alt="animation of classical max pooling">

The last column may not be used depending on the size of the kernel/input and stride value. 
To avoid this, pooling layers with adaptive padding `PadMaxPool3d` were implemented in `clinicadl` to exploit information from the whole feature map.

<img src="https://drive.google.com/uc?id=14R_LCTiV0N6ZXm-3wQCj_Gtc1LsXdQq_" style="height: 200px;" alt="animation of max pooling with adaptive pooling">

!!! note "Adapt the padding... or the input!"
    To avoid this problem, deep learners often choose to resize their input to have sizes 
    equal to 2<sup>n</sup> with maxpooling layers of size and stride of 2.

## Autoencoders construction from CNN architectures

In `clinicadl`, an autoencoder is derived from a CNN architecture:

- the encoder corresponds to the convolutional part of the CNN,
- the decoder is composed of the transposed version of the operations used in the encoder.

![Illustration of a CNN and the corresponding autoencoder](../images/transfer_learning.png)

The list of the transposed version of the modules can be found below:

- `Conv3d` → `ConvTranspose3d`
- `PadMaxPool3d` → `CropMaxUnpool3d` 
(specific module of `clinicadl` used to reconstruct the feature map produced by pooling layers with adaptive padding)
- `MaxPool3d` → `MaxUnpool3d`
- `Linear` → `Linear` with an inversion in `in_features` and `out_features`,
- `Flatten` → `Reshape`
- `LeakyReLU` → `LeakyReLU` with the inverse value of alpha,
- other → copy of itself

## Transfer learning

It is possible to transfer trainable parameters between models. In the following list the weights are transferred from `source task` to `target task`:

- `autoencoder` to `cnn`: The trainable parameters of the convolutional part of the `cnn` 
(convolutions and batch normalization layers) take the values of the trainable parameters of the encoder part of the source autoencoder.
- `cnn` to `cnn`: All the trainable parameters are transferred between the two models.
- `autoencoder` to `multicnn`: The convolutional part of each CNN of the `multicnn` run is initialized
 with the weights of the encoder of the source autoencoder.
- `cnn` to `multicnn`: Each CNN of the `multicnn` run is initialized with the weights of the source CNN.
- `multicnn` to `multicnn`: Each CNN is initialized with the weights of the corresponding one in the source experiment.

## Optimization

The optimizer used in `clinicadl train` is [Adam](https://arxiv.org/abs/1412.6980). 

Usually, the optimizer updates the weights after one iteration, an iteration corresponding 
to the processing of one batch of images.
In ClinicaDL, it is possible to accumulate the gradients with `accumulation_steps` during `N` iterations to update
the weights of the network every `N` iterations. This allows simulating a larger batch size
even though the computational resources are not powerful enough to allow it.

<p style="text-align: center;">
<code>virtual_batch_size</code> = <code>batch_size</code> * <code>accumulation_steps</code>
</p>

## Evaluation

In some frameworks, the training loss may be approximated using the sum of the losses of the last
batches of data seen by the network. In ClinicaDL, set (train or validation) performance is always evaluated
on all the images of the set.

By default during training, the network performance on train and validation is evaluated at the end of each epoch.
It is possible to perform inner epoch evaluations by setting the value of `evaluation_steps` to the number of 
weight updates before evaluation. Inner epoch evaluations allow better evaluating the progression of the network
during training. 

!!! warning "Computation time"
    Setting `evaluation_steps` to a small value may considerably increase computation time.

## Model selection

The selection of a model is associated to a metric evaluated on the validation set:

- autoencoders are selected based on the loss (mean squared error),
- CNNs are selected based on the balanced accuracy and the loss (cross-entropy loss).

At the end of each epoch, if the validation performance of the current state is better than the best one ever seen, 
the current state of the model is saved in the corresponding best model folder.
Such comparison and serialization is only performed at the end of an epoch, even though inner epoch evaluations 
are performed.

## Stopping criterion

By default, early stopping is enabled to save computation time. This method automatically stops training
if during `patience` epochs, the validation loss at the end of an epoch never became smaller than the best validation
loss ever seen * (1 - `tolerance`). Early stopping can be disabled by setting `patience` to `0`.

If early stopping is disabled, or if its stopping criterion was never reached, training stops when the maximum number
of epochs `epochs` is reached.

## Soft voting

<SCRIPT SRC='https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'></SCRIPT>
<SCRIPT>MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}})</SCRIPT> 

For classification tasks that take as input a part of the MRI volume (*patch, roi or slice*), 
an ensemble operation is needed to obtain the label at the image level.

For example, size and stride of 50 voxels on linear preprocessing leads to the classification of 36 patches,
but they are not all equally meaningful.
Patches that are in the corners of the image are mainly composed of background and skull and may be misleading,
whereas patches within the brain may be more useful.

![Comparison of meaningful and misleading patches](../images/patches.png)

Then the image-level probability of AD *p<sup>AD</sup>* will be:

$$ p^{AD} = {\sum_{i=0}^{35} bacc_i * p_i^{AD}}$$

where:

- *p<sub>i</sub><sup>AD</sup>* is the probability of AD for patch *i*,
- *bacc<sub>i</sub>* is the validation balanced accuracy for patch *i*.

## Multi-cohort

Starting from version 0.2.1, it is possible to use ClinicaDL's functions on several datasets at the same time.
In this case, the `multi-cohort` flag must be given, and the `caps_directory` and the `tsv_path`
correspond to TSV files.

The `caps_directory` variable must lead to a TSV file with two columns:
- `cohort` the name of the cohort (must correspond to the values in `tsv_path`),
- `path` the path to the corresponding [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.

The `tsv_path` variable points to a TSV file with two columns:
- `cohort` the name of the cohort (must correspond to the values in `caps_directory`),
- `path` the path to the corresponding labels list, outputs of [`split`](../TSVTools.md#split---single-split-observing-similar-age-and-sex-distributions) 
or [`kfold`](../TSVTools.md#kfold---k-fold-split) methods.
- `diagnoses` the diagnoses that will be used in the cohort. Must correspond to a single string with labels accepted by
`clinicadl train` (`AD`, `BV`, `CN`, `MCI`, `sMCI` or `pMCI`) separated by commas.
See the [dedicated section](./Custom.md#custom-labels) to use custom labels.
