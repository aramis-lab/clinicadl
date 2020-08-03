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

#### Installation of clinicadl


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
                     {generate,preprocessing,extract,train,classify,tsvtool} ...

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
