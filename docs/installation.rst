.. _installation:

Installation
============

ClinicaDL 2.0 beta version can be installed by cloning the
`ClinicaDL project <https://github.com/aramis-lab/clinicadl>`_ and going to the ``clinicadl_v2`` branch. 
As it is a beta version, you are strongly encouraged to contribute. We thus suggest that you install
ClinicaDL in "developer mode" by `forking <https://docs.github.com/fr/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_
the repository.

Once it is forked, clone it and checkout to ``clinicadl_v2`` branch::

    git clone https://github.com/<your_github_user>/clinicadl.git
    git checkout clinicadl_v2

Then create your conda environment::

    conda create --name clinicadl_beta python=3.12
    conda activate clinicadl_beta

And install the dependencies using ``poetry``::

    cd clinicadl
    poetry install

.. dropdown:: Install ``poetry``

    To install ``poetry``, use ``pipx``::

        pipx install poetry

    To install pipx on ``macOS``::

        brew install pipx
        pipx ensurepath

    Otherwise, install via ``pip``::

        python3 -m pip install --user pipx
        python3 -m pipx ensurepath
