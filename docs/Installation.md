# Installation

You will find below the steps for installing `clinicadl` on Linux or Mac.
Please do not hesitate to contact us on the
[forum](https://groups.google.com/forum/#!forum/clinica-user) or
[GitHub](https://github.com/aramis-lab/AD-DL/issues)
if you encounter any issues.


## Quick start

### Python environment
You will need a Python environment to run Clinica. We advise you to
use [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Miniconda allows you to install, run, and update Python packages and their
dependencies. It can also create environments to isolate your libraries.
To install Miniconda, open a new terminal and type the following commands:

- If you are on Linux:
```{.sourceCode .bash}
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda-installer.sh
bash /tmp/miniconda-installer.sh
```

- If you are on Mac:
```{.sourceCode .bash}
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o /tmp/miniconda-installer.sh
bash /tmp/miniconda-installer.sh
```

#### Installation of ClinicaDL

The latest release of Clinica can be installed using `pip` as follows:

```bash
conda create --name clinicaEnv python=3.6
conda activate clinicaEnv
pip install clinica
```

### Installation of the third-party software packages
Depending on the pipeline that you want to use, you need to install
**pipeline-specific interfaces**. Not all the dependencies are necessary to run
Clinica.
Please refer to [this section](../Third-party) to determine which third-party
libraries you need to install.


### Running the Clinica environment
#### Activation of the Clinica environment

Now that you have created the Clinica environment, you can activate it:

```bash
conda activate clinicaEnv
activate-global-python-argcomplete --user #Only the first time you activate the environment
eval "$(register-python-argcomplete clinica)"
```

!!! success
    Congratulations, you have installed Clinica! At this point, you can try the
    basic `clinica` command and get the help screen:
    ```bash
    (clinicaEnv)$ clinica
    usage: clinica [-v] [-l file.log]  ...

    clinica expects one of the following keywords:

        run                 To run pipelines on BIDS/CAPS datasets.
        convert             To convert unorganized datasets into a BIDS hierarchy.
        iotools             Tools to handle BIDS/CAPS datasets.
        visualize           To visualize outputs of Clinica pipelines.
        generate            To generate pre-filled files when creating new
                            pipelines (for developers).

    Optional arguments:
      -v, --verbose         Verbose: print all messages to the console
      -l file.log, --logname file.log
                            Define the log file name (default: clinica.log)
    ```

    If you have successfully installed the third-party software packages, you are ready
    to run any of the pipelines proposed by Clinica.

    You can now learn how to [interact with Clinica](../InteractingWithClinica).

#### Deactivation of the Clinica environment
At the end of your session, remember to deactivate your Conda environment:
```bash
conda deactivate
```

## Developer installation

If you plan to contribute to Clinica or if you want to have the current development
version, you can either:

* Download the tarball for a specific version from our
[repository](https://github.com/aramis-lab/clinica/releases).
Then decompress it.
* Clone Clinica's repository from GitHub:
```bash
git clone https://github.com/aramis-lab/clinica.git
```

We suggest creating a custom Conda environment and installing Clinica using the
provided YML file:

```bash
cd clinica
conda env create -f environment.yml
```

By default, the environment is named `clinica_env`. You can choose a different
name by adding the option `--name my_clinica_environment`.

Clinica is installed within the environment created. Remember to
activate the environment before proceeding:

```bash
conda activate clinica_env
pip install -e . # Only the first time you activate the environment
activate-global-python-argcomplete --user # Only the first time you activate the environment
eval "$(register-python-argcomplete clinica)"
```

If everything goes well, type `clinica` and you should see the help message which
is displayed above.

At the end of your session, you can deactivate your Conda environment:
```bash
conda deactivate
```

Remember that Clinica will be only available inside your Conda environment.
<!-- Further information for Clinica's contributors can be found
[here](../CodingForClinica). -->

!!! warning  "In case your face `ResolvePackageNotFound: python==3.6` error"
    When installing Clinica with `conda`, you may see the following error:
    ```
    Collecting package metadata (repodata.json): done
    Solving environment: failed

    ResolvePackageNotFound:
    - python==3.6
    ```
    This is caused by newer version of conda. This has been corrected recently in the `dev` branch on Clinica.




We recommend to use `conda` or `virtualenv` to  install clinicadl inside a
python environment. E.g.,

```{.sourceCode .bash}
conda create --name clinicadl_env python=3.7
conda activate clinicadl_env
pip install numpy==1.17
pip install clinicadl
```

You can also install the developer version from the repository:
the active conda environment:

```{.sourceCode .bash}
conda create --name clinicadl_env python=3.7
conda activate clinicadl_env
git clone git@github.com:aramis-lab/AD-DL.git
cd AD-DL
cd clinicadl
pip install numpy==1.17
pip install -e .
```

### Running the clinicadl environment
#### Activation of the clinicadl environment

Now that you have created the clinicadl environment, you can activate it:

```bash
conda activate clinicadl_env
```

!!! success
    Congratulations, you have installed ClinicaDL! At this point, you can try the
    basic `clinicadl -h` command and get the help screen:
    ```Text
    (clinicadl_env)$ clinicadl -h
    usage: clinicadl [-h] [--verbose]
                     {generate,preprocessing,extract,quality_check,train,classify,tsvtool}
                     ...

    Deep learning software for neuroimaging datasets

    optional arguments:
      -h, --help            show this help message and exit
      --verbose, -v

    Task to execute with clinicadl::
      What kind of task do you want to use with clinicadl? (tsvtool,
      preprocessing, extract, generate, train, validate, classify).

      {generate,preprocessing,extract,quality_check,train,classify,tsvtool}
                            ****** Tasks proposed by clinicadl ******
        generate            Generate synthetic data for functional tests.
        preprocessing       Prepare data for training (needs clinica installed).
        extract             Create data (slices or patches) for training.
        quality_check       Performs quality check procedure for t1-linear
                            pipeline.Original code can be found at
                            https://github.com/vfonov/deep-qc
        train               Train with your data and create a model.
        classify            Classify one image or a list of images with your
                            previously trained model.
        tsvtool             Handle tsv files for metadata processing and data
                            splits
    ```


#### Deactivation of the ClinicaDL environment
At the end of your session, remember to deactivate your Conda environment:
```bash
conda deactivate
```


## Testing

Be sure to have the `pytest` library in order to run the test suite.
This test suite includes unit testing to be launched using the command line.

### Unit testing (WIP)

The CLI (command line interface) part is tested using `pytest`. We are planning
to provide unit tests for other tasks in the future. If you want to run
successfully the CLI tests you can use the following command line:

```{.sourceCode .bash}
pytest clinicadl/tests/test_cli.py
```

### Functional testing

Metadata processing and classification tasks can be tested.
To run these tests, go to the test folder and type the following
commands in the terminal:

```{.sourceCode .bash}
pytest ./test_classify.py
pytest ./test_tsvtool.py
```

!!! warning
    Data for testing is not currently provided,
    but release of anonymized datasets for testing is planned for future versions.
