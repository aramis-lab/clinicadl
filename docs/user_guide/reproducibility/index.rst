.. _user_guide_reproducibility:

3. Experiment management and reproducibility
=============================================

Two of ClinicaDL's guiding principles are **experiment management** — keeping every
output of an experiment together — and **reproducibility** — being able to run the
same experiment again and obtain the same result. This chapter presents the two
features that make this possible.

#. :doc:`Configuration classes <config>` — serialisable objects that record the
   parameters of the objects you use, so that an experiment can be saved and rebuilt.
#. :doc:`MAPS <maps>` — the single folder in which a
   :py:class:`~clinicadl.train.Trainer` gathers all the outputs and hyperparameters of
   an experiment.

.. toctree::
   :maxdepth: 2
   :hidden:

   config
   maps
