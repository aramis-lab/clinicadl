.. _user_guide_workflow_model:

2.1 Defining a model
====================

In ClinicaDL, a **model** is more than a neural network. A
:py:class:`~clinicadl.models.Model` bundles together everything needed to train and
evaluate a network: the network itself, a loss function, an optimizer, and the logic
that defines how a batch flows forward, how gradients are computed, how the weights
are optimized, and how the model is evaluated. By gathering this logic in one object, ClinicaDL can offer a generic
:py:class:`~clinicadl.train.Trainer` that works with any model.

The Model class
---------------

Every model in ClinicaDL inherits from the base :py:class:`~clinicadl.models.Model`,
which is itself a :py:class:`torch.nn.Module`. ``Model`` defines the interface that the
:py:class:`~clinicadl.train.Trainer` relies on — a handful of methods that capture the
**essential logic** of an experiment:

- :py:meth:`~clinicadl.models.Model.forward_step` — how a batch is passed forward and
  the loss computed;
- :py:meth:`~clinicadl.models.Model.backward_step` and
  :py:meth:`~clinicadl.models.Model.optimization_step` — how gradients are computed and
  applied;
- :py:meth:`~clinicadl.models.Model.evaluation_step` — how inference is performed during model evaluation;
- :py:meth:`~clinicadl.models.Model.build_optimizers` and
  :py:meth:`~clinicadl.models.Model.get_loss_functions` — how the optimizers and loss
  functions are built.

You will rarely implement all of this yourself. ClinicaDL ships two ready-to-use
models that already define this logic for the most common cases —
:py:class:`~clinicadl.models.SupervisedModel` and
:py:class:`~clinicadl.models.ReconstructionModel`, described below. When you need a
different behaviour, you can subclass one of them (or ``Model`` itself) and override only
the relevant method, as covered in :doc:`Chapter 4 <../customising/index>`.

The SupervisedModel
-------------------

:py:class:`~clinicadl.models.SupervisedModel` is the model for usual **supervised**
tasks — classification, regression and segmentation. You give it a network, a loss
and an optimizer, and tell it which field of your :py:class:`Samples <clinicadl.data.structures.Sample>`
holds the label:

.. code-block:: python

    import torch
    from clinicadl.models import SupervisedModel
    from clinicadl.networks.nn import CNN
    from clinicadl.optim.optimizers.config import AdamConfig

    model = SupervisedModel(
        network=CNN(
            in_shape=(1, 169, 208, 179),
            num_outputs=2,
            conv_args={"channels": [8, 16, 32]},
        ),
        loss=torch.nn.CrossEntropyLoss(),     # any PyTorch-style loss
        optimizer=AdamConfig(),
        label_key="diagnosis",                # the label field in the Sample
    )

Three ingredients deserve a closer look:

``network``
    The neural network. You can pass any :py:class:`torch.nn.Module` — one of the
    architectures from :py:mod:`clinicadl.networks.nn` (see :ref:`below <user_guide_workflow_networks>`)
    or your own — or a :py:mod:`network configuration object <clinicadl.networks.config>`.

``loss``
    The loss function. Any **PyTorch-style** loss works: a callable returning a
    one-item :py:class:`torch.Tensor` and exposing a ``reduction`` attribute that can
    be set to ``"none"``. This includes the losses of :py:mod:`torch.nn`, the losses of
    :monai:`MONAI <losses.html#loss-functions>`, your own, or a :py:mod:`loss configuration object <clinicadl.losses.config>`.

``optimizer``
    The optimizer, passed as a
    :py:mod:`configuration object <clinicadl.optim.optimizers.config>` such as
    :py:class:`~clinicadl.optim.optimizers.config.AdamConfig` or
    :py:class:`~clinicadl.optim.optimizers.config.SGDConfig`.

.. note::

    Losses and networks can be passed as raw objects, but the optimizer is always
    passed as a *configuration object* here. Configuration classes — which record an
    object's parameters in a serialisable, reproducible form — are the subject of
    :doc:`Chapter 3 <../reproducibility/index>`.

By default, a ``SupervisedModel`` passes the whole image through the network during inference. To run
inference patch-by-patch or slice-by-slice instead, pass an
:py:class:`~clinicadl.infer.Inferer` via the ``inferer`` argument — this is covered
in :doc:`Evaluating <evaluating>`.

The ReconstructionModel
-----------------------

:py:class:`~clinicadl.models.ReconstructionModel` is the counterpart for **image
reconstruction**, e.g. with an autoencoder. It works just like a
``SupervisedModel``, except that the loss compares the network's output to the input
image — so there is no label to specify:

.. code-block:: python

    import torch
    from clinicadl.models import ReconstructionModel
    from clinicadl.networks.nn import AutoEncoder
    from clinicadl.optim.optimizers.config import AdamConfig

    model = ReconstructionModel(
        network=AutoEncoder(
            in_shape=(1, 80, 96, 80),
            latent_size=128,
            conv_args={"channels": [8, 16, 32]},
        ),
        loss=torch.nn.MSELoss(),
        optimizer=AdamConfig(),
    )

.. _user_guide_workflow_networks:

2.1.1 Neural networks
---------------------

:py:mod:`clinicadl.networks.nn` provides a catalogue of neural networks, all
subclasses of :py:class:`torch.nn.Module`, organised in three families:

**Builders** — generic, fully configurable networks you assemble from your own
specifications:

- :py:class:`~clinicadl.networks.nn.MLP` — a multilayer perceptron;
- :py:class:`~clinicadl.networks.nn.ConvEncoder` / :py:class:`~clinicadl.networks.nn.ConvDecoder`
  — convolutional encoders and decoders;
- :py:class:`~clinicadl.networks.nn.CNN` — a convolutional encoder followed by a MLP;
- :py:class:`~clinicadl.networks.nn.Generator` — an MLP followed by a convolutional decoder (the symmetric of ``CNN``);
- :py:class:`~clinicadl.networks.nn.AutoEncoder` and :py:class:`~clinicadl.networks.nn.VAE`.

**Common architectures** — well-known networks, configurable in their depth and width:

- :py:class:`~clinicadl.networks.nn.UNet`, :py:class:`~clinicadl.networks.nn.AttentionUNet`;
- :py:class:`~clinicadl.networks.nn.DenseNet`, :py:class:`~clinicadl.networks.nn.ResNet`,
  :py:class:`~clinicadl.networks.nn.SEResNet`, :py:class:`~clinicadl.networks.nn.ViT`.

**Literature variants** — ready-to-use architectures with the exact settings from
their original papers, for instance
:py:class:`~clinicadl.networks.nn.ResNet18` … :py:class:`~clinicadl.networks.nn.ResNet152`,
:py:class:`~clinicadl.networks.nn.DenseNet121` … :py:class:`~clinicadl.networks.nn.DenseNet201`,
or :py:class:`~clinicadl.networks.nn.ViTB16`.

.. code-block:: python

    from clinicadl.networks.nn import ConvEncoder, ResNet18

    # a builder: you specify the architecture
    encoder = ConvEncoder(spatial_dims=3, in_channels=1, channels=[8, 16, 32])

    # a literature variant: ready to use
    resnet = ResNet18(num_outputs=2)

Each network has a matching configuration class in
:py:mod:`clinicadl.networks.config` (e.g.
:py:class:`~clinicadl.networks.config.ResNet18Config`), so a network can also be
described in a serialisable way — see :doc:`Chapter 3 <../reproducibility/index>`.

----

With a model in hand, you are ready to :doc:`train it <training>`.
