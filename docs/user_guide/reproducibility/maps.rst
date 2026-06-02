.. _user_guide_reproducibility_maps:

3.2 MAPS
========

A **MAPS** (Model Analysis and Processing Structure) is the cornerstone of experiment
management in ClinicaDL: a single directory that holds *everything* about an
experiment. It is represented in Python by the :py:class:`~clinicadl.io.maps.Maps`
class.

What a MAPS contains
--------------------

Whenever you train, validate or test a model, the
:py:class:`~clinicadl.train.Trainer` writes to its MAPS:

- **Outputs of training and evaluation** — the trained network weights and
  checkpoints, the computed metrics, the logs, computational statistics, and the
  results of testing and inference on new data.
- **All the hyperparameters used** — the model, the optimization settings, the
  callbacks, the metrics and the data splits, stored as
  :doc:`configuration classes <config>`.

Because everything lives in one place, a MAPS serves the two goals of this chapter at
once:

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: :fas:`share-nodes` Experiment management

        One self-contained, **shareable** folder fully describes one experiment —
        results, logs and configuration together. No more scattered output files.

    .. grid-item-card:: :fas:`rotate` Reproducibility

        Since the MAPS records the configuration of every object used, an experiment
        can be **rebuilt and run again** from the folder alone.

Creating and reopening a MAPS
-----------------------------

You rarely create a MAPS by hand: a :py:class:`~clinicadl.train.Trainer` creates one
for you from the path you give it (see :doc:`Chapter 2 <../workflow/index>`).

.. code-block:: python

    from clinicadl.train import Trainer

    trainer = Trainer(maps="maps", model=model)   # creates the MAPS at ./maps
    trainer.train(split)

Later — in another session, or to share your results — you can reopen the experiment.
:py:meth:`Trainer.from_maps <clinicadl.train.Trainer.from_maps>` rebuilds the whole
trainer from the configurations stored in the MAPS:

.. code-block:: python

    trainer = Trainer.from_maps("maps")

To inspect the folder directly, open it as a :py:class:`~clinicadl.io.maps.Maps`,
which exposes the path of every file and subdirectory it contains:

.. code-block:: python

    from clinicadl.io.maps import Maps

    maps = Maps("maps")

.. warning::

    Reproducibility relies on :doc:`configuration classes <config>`. If you used a
    **raw object** that has no configuration class (a custom transform, an external
    network, …), the MAPS will still run, but ClinicaDL cannot guarantee that this
    object is reproduced exactly — and :py:meth:`Trainer.from_maps <clinicadl.train.Trainer.from_maps>`
    may not be able to rebuild it. Prefer configuration classes whenever one exists.

----

This closes Chapter 3. The :doc:`final chapter <../customising/index>` shows how to
go beyond the ready-made objects and tailor ClinicaDL to your own needs.
