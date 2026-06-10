.. _user_guide:

User Guide
==========

Welcome to the ClinicaDL User Guide. This guide walks you through the library, from manipulating
your neuroimaging data to building and managing a full deep learning experiment.

If you are new to ClinicaDL, start with the :doc:`Quickstart <../quickstart>`:
it explains what ClinicaDL is for and offers a condensed, end-to-end tour of the
library. The following chapters then go into the details, one building block at a
time.

.. grid::

    .. grid-item::
        :columns: 12 12 6 6
        :margin: 2 2 0 0

        .. card:: :fas:`brain` 1. Manipulating neuroimaging data
            :link: data/index
            :link-type: doc
            :class-card: sd-shadow-md sd-h-100
            :class-title: sd-text-primary
            :text-align: center

            Data structures, BIDS datasets, transforms, splits and batching

    .. grid-item::
        :columns: 12 12 6 6
        :margin: 2 2 0 0

        .. card:: :fas:`diagram-project` 2. Building a deep learning workflow
            :link: workflow/index
            :link-type: doc
            :class-card: sd-shadow-md sd-h-100
            :class-title: sd-text-primary
            :text-align: center

            Defining a model, training, evaluating and callbacks

    .. grid-item::
        :columns: 12 12 6 6
        :margin: 2 2 0 0

        .. card:: :fas:`box-archive` 3. Experiment management and reproducibility
            :link: reproducibility/index
            :link-type: doc
            :class-card: sd-shadow-md sd-h-100
            :class-title: sd-text-primary
            :text-align: center

            Configuration classes and the MAPS

    .. grid-item::
        :columns: 12 12 6 6
        :margin: 2 2 0 0

        .. card:: :fas:`screwdriver-wrench` 4. Customising your experiment
            :link: customising/index
            :link-type: doc
            :class-card: sd-shadow-md sd-h-100
            :class-title: sd-text-primary
            :text-align: center

            Extend ClinicaDL with your own objects

.. toctree::
   :maxdepth: 2
   :hidden:

   data/index
   workflow/index
   reproducibility/index
   customising/index
