.. _user_guide_workflow:

2. Building a deep learning workflow
====================================

Once your data is ready (see :doc:`Chapter 1 <../data/index>`), you can build the
actual deep learning workflow: define a model, train it, evaluate it, and customise
the training loop. ClinicaDL hides the repetitive PyTorch plumbing while leaving the
essential logic — the one that defines your experiment — in your hands.

This chapter follows the life cycle of a model:

#. :doc:`Defining a model <model>` — wrap a neural network, a loss and an optimizer
   into a :py:class:`~clinicadl.models.Model`, and pick an architecture from
   :py:mod:`clinicadl.networks.nn`.
#. :doc:`Training <training>` — run the training loop with a
   :py:class:`~clinicadl.train.Trainer`, and resume it if it gets interrupted.
#. :doc:`Evaluating <evaluating>` — compute metrics on validation and test data, and
   customise the inference step.
#. :doc:`Callbacks <callbacks>` — plug non-essential logic (logging, early stopping,
   checkpointing, …) into the training loop.

.. toctree::
   :maxdepth: 2
   :hidden:

   model
   training
   evaluating
   callbacks
