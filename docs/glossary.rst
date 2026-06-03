.. _glossary:

Glossary
========

.. glossary::
    :sorted:

    **AMP**
        Automatic Mixed Precision — a training technique that combines 16- and 32-bit
        floating-point operations to speed up computation and reduce memory usage.
        See :py:mod:`torch.amp`.

    **BIDS**
        Brain Imaging Data Structure — a community standard for naming and organising
        neuroimaging files on disk (`source <https://bids.neuroimaging.io/index.html>`__).

    **BIDS derivative**
        A dataset derived from a raw :term:`BIDS` dataset (e.g. example preprocessed
        images) that follows the
        :bids:`BIDS derivatives convention <derivatives/introduction.html>`.

    **CAPS**
        ClinicA Processed Data Structure — the output format of :clinica:`Clinica <>`'s
        preprocessing pipelines, organised as a :term:`BIDS derivative`
        (:clinica:`source <CAPS/Introduction>`).

    **MAPS**
        Model Analysis and Processing Structure — the single directory in which ClinicaDL
        gathers all the outputs and hyperparameters of an experiment. See
        :ref:`Section 3.2 <user_guide_reproducibility_maps>`.

    **RAS+**
        A standard anatomical orientation convention for neuroimaging. The
        axes are respectively: left to Right, posterior to Anterior, and inferior to Superior
        (:nibabel:`source <coordinate_systems.html#naming-reference-spaces>`).

    **data leakage**
        A flaw in the evaluation of a model where information from the test data leaks
        into training, leading to over-optimistic performance estimates. In neuroimaging,
        it typically happens when different sessions of the same participant are split
        across the training and test sets.
