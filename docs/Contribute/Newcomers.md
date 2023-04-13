# Developer installation

If you plan to contribute to ClinicaDL, we suggest you 
[create a fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) 
of [ClinicaDL repo](https://github.com/aramis-lab/clinicadl).

Then clone your fork from GitHub:
```{.sourceCode .bash}
git clone https://github.com/<your_name>/clinicadl.git
```

Once you cloned the repository in your personal folder on the lustre, move in it and install 
the latest version of poetry using pipx:
```{.sourceCode .bash}
cd clinicadl
pipx install poetry
```

To install pipx on macOS:
```{.sourceCode .bash}
brew install pipx
pipx ensurepath
```
Otherwise, install via pip (requires pip 19.0 or later):
```{.sourceCode .bash}
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

We suggest creating a custom Conda environment for your fork, so you can
test your modifications, and install all the dependencies inside your environment using poetry:

```{.sourceCode .bash}
conda env create -f environment.yml --name clinicadl_dev
conda activate clinicadl_dev
poetry install
```

If everything goes well, type `clinicadl -h` and you should see the help message of ClinicaDL.

At the end of your session, you can deactivate your Conda environment:
```bash
conda deactivate
```

# Suggest modifications

Please suggest modifications based on the last version of the `dev` branch of the original repo.
The first time, add the remote URL `upstream` by running the following command in your fork's repository:
```bash
git remote add upstream https://github.com/aramis-lab/clinicadl.git
```

You can check which remote URLs are linked to your repository with the following command:
```bash
git remote -v
```

To create a new branch up-to-date, you will need to fetch and pull the modifications from the original repo:
```bash
git checkout dev
git fetch upstream
git pull upstream dev
```
You can now create a new branch from your up-to-date `dev` branch:
```bash
git checkout -b <branch_name>
```

You can open a [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) 
as soon as you want on the original repo, to inform / ask advice from the admins.
If you did not run any tests (see next section), you can first open a draft pull request to avoid running the continuous integration
on your code.

# Add documentation

Documentation of ClinicaDL is deployed by [readthedocs](https://readthedocs.org/).
Source files are available in the `docs` folder of the repo.

To build the documentation locally, please install the specific requirements:
```bash
pip install -r docs/requirements.txt
```

Once it is done you can run `mkdocs serve` at the root of the repo and copy-paste
the URL in your terminal in a web browser to browse the documentation.
