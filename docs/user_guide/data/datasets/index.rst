.. _user_guide_data_bids:

1.2. Reading neuroimaging datasets
==================================

We strongly encourage users to follow the :term:`BIDS` standard when organizing their data.
ClinicaDL provides several tools to handle one or multiple BIDS datasets,
even though it also allows users to implement their own data-reading logic for loading non-BIDS-compliant datasets.

#. :doc:`BIDS datasets <bids>` — basic objects to read BIDS-compliant datasets.
#. :doc:`Advanced tips <advanced>` —  speeding up data loading, joining multiple datasets,
   and reading non-BIDS-compliant datasets.

.. toctree::
   :maxdepth: 1
   :hidden:

   bids
   advanced