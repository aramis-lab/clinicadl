.. _contributing:

Contributing
============

There are many ways in which you can contribute to the ongoing development of
ClinicaDL. For example, you can:

- `Ask a question <https://github.com/aramis-lab/clinicadl/discussions/new?category=q-a>`_
- `Report a bug <https://github.com/aramis-lab/clinicadl/issues/new>`_
- `Submit a bug fix <https://github.com/aramis-lab/clinicadl/compare>`_
- `Propose a new feature <https://github.com/aramis-lab/clinicadl/discussions/new?category=ideas>`_
- `Discuss the current state of the code <https://github.com/aramis-lab/clinicadl/discussions/new/choose>`_

Making direct contributions
---------------------------

Open an issue on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~

The very first thing to do is to open an issue on GitHub, in which you describe the
bug or the feature you want to work on.

By doing this, you will get immediate feedback from the developer team on the
contribution you are planning to make (is it already being addressed by someone else,
is it useful and within the scope of the project, etc.).

This preliminary discussion will save you a lot of time, as the developer team will
also be able to point you to the right place to modify the code.

Set up your local environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the issue has been discussed and you are ready to implement your contribution,
you need to get a copy of the source code on which you can make edits.

1. Fork the repository
^^^^^^^^^^^^^^^^^^^^^^

We use a fairly standard process in the open source world, where contributors fork
the project and work on their fork before making a Pull Request to the upstream
repository. If this is your first time contributing to an open source project, please
read this very clear `tutorial <https://github.com/firstcontributions/first-contributions>`_.

To sum up these steps:

- Go to the `repository <https://github.com/aramis-lab/clinicadl>`_ and click the
  "Fork" button to create your own copy of the project.
- Clone your fork to your local computer, where ``<username>`` is your GitHub
  username:

  .. code-block:: bash

    git clone git@github.com:<username>/clinicadl.git

- Create a branch for the feature you want to work on. Since the branch name will
  appear on the pull request, use a meaningful name (e.g. ``fix-something-important``):

  .. code-block:: bash

    git checkout -b <branch_name>

2. Set up a Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ClinicaDL uses `Poetry <https://python-poetry.org/>`_ to manage its dependencies and
its packaging. From the root of the cloned repository, install ClinicaDL together
with its development dependencies in an isolated environment:

.. code-block:: bash

  poetry install --with dev

This installs the library in editable mode, so your edits to the source code are
picked up immediately.

.. note::
  Poetry will automatically create a virtual Python environment. But you can
  also run this command in your dedicated ``conda`` environment if you have created one.

1. Set up the pre-commit hook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To keep the code style consistent across ClinicaDL, we rely on a set of tools
(a formatter, a linter, etc.) run automatically through a
`pre-commit <https://pre-commit.com/>`_ hook. Install it once, inside the cloned
folder:

.. code-block:: bash

    pre-commit install

From then on, the tools run automatically every time you commit. See
`Coding style conventions`_ for the details.

Make the modifications
~~~~~~~~~~~~~~~~~~~~~~

Your environment is set. Make the modifications you planned to make:

- Commit locally as you progress (``git add`` and ``git commit``).
- Push your changes to your fork on GitHub:

  .. code-block:: bash

    git push origin <branch_name>

Test your modifications
~~~~~~~~~~~~~~~~~~~~~~~

At this point you are happy with your contribution and you want to merge it into the
official source code of ClinicaDL.

A first thing to verify is whether your changes broke some unit tests. You can run the
unit-test suite with:

.. code-block:: bash

    make unit-tests

If you have worked on GPU features, run the following command as well (on a machine with a GPU):

.. code-block:: bash

    make gpu-unit-tests

Similarly, if you have worked on multi-GPU features, run on a multi-GPU machine:

.. code-block:: bash

    make multi-gpu-unit-tests

These commands run the entire suite of unit tests (it should take a few minutes) and tells you
how many passed and how many failed. If all tests pass, you can proceed with :ref:`modifying
the documentation <documentation>`. 
If some fail, inspect which ones and whether this is expected.

Sometimes a fix or a new feature legitimately changes existing behaviour and the
corresponding tests need to be updated or new tests must be written (see :ref:`Tests <tests>`).

.. note::
  No problem if you don’t have access to a GPU machine. Just open a Pull Request,
  and the tests will run automatically on a suitable machine.

.. _documentation:

Modify the documentation (if need be)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your contribution involves changes to the public documentation
(for example, adding a new object that needs to be documented),
make sure the documentation :ref:`builds correctly <building_documentation>` after your modifications.

.. admonition:: How do I know if the documentation should be modified?
  :class: tip

  Discuss it with the reviewer! A few hints:

  - If you modify the signature or the behavior of a documented class or function,
    you should probably modify its docstring. A modification in the docstring will
    automatically appear in the public documentation.
  - If you add a new object that should be documented, mention it somewhere
    in ``docs/api``, and potentially in ``docs/user_guide``.

.. _pull_request:

Submit a Pull Request
~~~~~~~~~~~~~~~~~~~~~

Now that everything seems to work on your machine, you can submit a Pull Request!

To submit a Pull Request, go to your fork on GitHub: the new branch will show up with
a green "Pull Request" button — click it. Open the Pull Request against the ``dev``
branch of ``aramis-lab/clinicadl``.

You then need to provide a title and a description, and both are important. The
description should clearly start with ``Closes #XXXX``, where ``XXXX`` is the number
of the issue you opened initially. If the issue already describes the context and what
needs to be done, that may be enough; otherwise, this is the right place to add any
information about your implementation.

The review process
~~~~~~~~~~~~~~~~~~~

Your Pull Request will then be reviewed. A friendly discussion with the reviewer(s)
will take place in order to improve the quality of your contribution if needed.

If so, update your Pull Request until your changes are approved. The Pull Request
updates automatically as you push to your branch, and it will finally be merged by the
reviewer(s).

Additional tips
---------------

Here are a few additional things that may come in handy while contributing.

.. _building_documentation:

.. dropdown:: Building the documentation

   The documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_ and the API Reference is generated
   automatically from the docstrings. The ``docs`` folder contains all the source files,
   written in `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.

   To build the documentation locally, install the documentation dependencies and run
   Sphinx:

   .. code-block:: bash

      poetry install --with docs
      cd docs
      make html

   The result is then available in ``docs/_build/html``. You can visualize it
   in a web browser at the address http://localhost:8888/ after running the following command:

   .. code-block:: bash

      python -m http.server 8888 --directory '_build/html'

   .. tip::
     To get started with documentation building using Sphinx, you can follow
     `this tutorial <https://www.aramislab.fr/tuto-doc/tutorial/index.html>`_.

.. dropdown:: Keeping your fork up to date

   If you plan to contribute regularly to ClinicaDL, you have to synchronize your fork
   regularly with the main repository:

   #. Add the main ClinicaDL repository as a remote (``upstream`` is a conventional name,
      but you can call it whatever you want):

      .. code-block:: bash

          git remote add upstream https://github.com/aramis-lab/clinicadl.git

   #. Fetch the data associated with that remote (this only downloads it):

      .. code-block:: bash

          git fetch upstream

   #. To get the latest version of the ``dev`` branch, switch to your own ``dev`` branch
      and pull from ``upstream``:

      .. code-block:: bash

          git checkout dev
          git pull upstream dev

   Now the ``dev`` branch of your fork is up to date with ``aramis-lab/clinicadl``.

.. dropdown:: Solving conflicts in a pull request

   It may happen that your Pull Request has conflicts: your branch tries to modify
   the same portion of code as a commit recently merged into ``dev``.

   You then have to integrate the latest commits of ``dev`` into your feature branch. We
   **rebase** rather than merge, so that the git history stays linear:

   #. Go to your feature branch and start an interactive rebase onto ``upstream/dev``:

      .. code-block:: bash

          git checkout <branch_name>
          git rebase -i upstream/dev

      Keep only the commits that belong to your branch, on top of ``upstream/dev``. The
      conflicts appear once you validate the rebasing.

   #. Solve the conflicts, following Git's instructions.

   #. Once it is done, force-push your branch (the history has been rewritten):

      .. code-block:: bash

          git push --force-with-lease origin <branch_name>

   Your Pull Request no longer has conflicts.

Coding style conventions
------------------------

ClinicaDL is a Python API that other people read and call, so clarity and consistency
matter as much as correctness:

- **Docstrings** follow the `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_
  style. Every public class, function and method should have a docstring. Whenever
  possible, include a short, runnable example (many existing docstrings use the example data structures provided 
  in :py:mod:`clinicadl.data.structures.examples`).
- **Type hints** are expected on public functions, methods and class attributes. They serve as useful documentation
  and help ensure code consistency through type checking.
- More generally, we follow the `PEP 8 <https://peps.python.org/pep-0008/>`_ convention.

The code style is enforced automatically by the `pre-commit <https://pre-commit.com/>`_
hook installed above, which runs `ruff <https://docs.astral.sh/ruff/>`_ (both the
linter and the formatter) and `codespell <https://github.com/codespell-project/codespell>`_.
You can also run all the checks manually on the whole code base:

.. code-block:: bash

    pre-commit run --all-files

.. _tests:

Tests
-----

ClinicaDL is tested with `pytest <https://docs.pytest.org/>`_. The tests live in two
places:

- ``tests/unittests`` — fast, isolated unit tests that check individual objects;
- ``tests/functional`` — end-to-end tests that exercise full workflows (these require
  reference data and are mostly run in continuous integration).

Most contributions only need the unit tests, which you run with:

.. code-block:: bash

    make unit-tests

License
-------

By contributing to ClinicaDL, you agree that your contributions will be licensed under
its `MIT License <https://github.com/aramis-lab/clinicadl/blob/dev/LICENSE.txt>`_.
