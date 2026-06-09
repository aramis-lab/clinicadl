.. _installation:

Installation
============

.. warning::
    ClinicaDL requires Python >=3.10, <=3.14

Install a released version
--------------------------

.. _setup_virtual_env:

1. Setup a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend that you install ClinicaDL in a virtual Python environment,
either managed with the standard library ``venv`` or with ``conda``
(see `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ for instance).
Either way, create and activate a new python environment.

With ``venv``:

.. code-block:: bash

    python3 -m venv /<path_to_new_env>
    source /<path_to_new_env>/bin/activate

Windows users should change the last line to ``\<path_to_new_env>\Scripts\activate.bat``
in order to activate their virtual environment.

With ``conda``:

.. code-block:: bash

    conda create -n clinicadl python=3.12
    conda activate clinicadl

2. Install ClinicaDL with pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following command in the proper Python environment:

.. code-block:: bash

    pip install clinicadl

Install in development mode
---------------------------

Find development setup instructions in the :ref:`contribution guide <setup_dev>`.
