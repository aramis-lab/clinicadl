.. _user_guide_workflow:

2. Building a deep learning workflow
====================================

Once your data is ready (see :doc:`Chapter 1 <../data/index>`), you can build the
actual deep learning workflow: define a model, train it, and evaluate it.
ClinicaDL abstracts away most of PyTorch's complexity while providing tools to modify
the default behavior and customise your training and validation logic.

#. :doc:`Defining a model <model>` — define a neural network, a loss and an optimizer,
   as well as your training and evaluation logic.
#. :doc:`Training <training>` — run the training loop, and resume it if it gets interrupted.
#. :doc:`Evaluating <evaluating>` — compute metrics on validation and test data.
#. :doc:`Callbacks <callbacks>` — plug non-essential logic (logging, early stopping,
   checkpointing, etc.) into the training loop.

.. toctree::
   :maxdepth: 2
   :hidden:

   model
   training
   evaluating
   callbacks
