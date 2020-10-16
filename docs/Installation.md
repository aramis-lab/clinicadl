# Installation

You will find below the steps for installing `clinicadl` on Linux or Mac.
Please do not hesitate to contact us on the
[forum](https://groups.google.com/forum/#!forum/clinica-user) or
[GitHub](https://github.com/aramis-lab/AD-DL/issues)
if you encounter any issues.

## Prepare your Python environment
You will need a Python environment to run ClinicaDL. We advise you to
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

## Install ClinicaDL

The latest release of ClinicaDL can be installed using `pip` as follows:

```{.sourceCode .bash}
conda create --name clinicadlEnv python=3.7
conda activate clinicadlEnv
pip install clinicadl
```

## Run the ClinicaDL environment
#### Activation of the ClinicaDL environment

Now that you have created the ClinicaDL environment, you can activate it:

```{.sourceCode .bash}
conda activate clinicadlEnv
```

!!! success
    Congratulations, you have installed ClinicaDL! At this point, you can try the
    basic `clinicadl -h` command and get the help screen:
    ```Text
    (clinicadlEnv)$ clinicadl -h
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


### Deactivation of the ClinicaDL environment
At the end of your session, remember to deactivate your Conda environment:
```{.sourceCode .bash}
conda deactivate
```


<!--## Developer installation

If you plan to contribute to ClinicaDL or if you want to have the current development
version, you can either:

* Download the tarball for a specific version from our
[repository](https://github.com/aramis-lab/AD-DL/releases).
Then decompress it.
* Clone ClinicaDL's repository from GitHub:
```{.sourceCode .bash}
git clone https://github.com/aramis-lab/AD-DL.git
```

We suggest creating a custom Conda environment and installing Clinica using the
provided YML file:

```{.sourceCode .bash}
conda create --name my_clinicadl_environment python=3.7
```

By default, the environment is named `clinica_env`. You can choose a different
name by adding the option .

Clinica is installed within the environment created. Remember to
activate the environment before proceeding:

```bash
conda activate my_clinicadl_environment
cd AD-DL/clinicadl
pip install -e .
```

If everything goes well, type `clinicadl -h` and you should see the help message which
is displayed above.

At the end of your session, you can deactivate your Conda environment:
```bash
conda deactivate
```

Remember that ClinicaDL will be only available inside your Conda environment.
-->

## Testing ClinicaDL

!!! warning
    Data for testing must be manually downloaded (see below). Be sure to install
    pytest & co (see [requirements-dev.txt](../requirements-dev.txt) file)
    inside your developement environment.

Main functionalities of ClinicaDL can be tested using the functions provided in
the `tests` folder (this folder is not included in the package but it can be
cloned from the main repository).

The tests run for every commit/PR in our Continuos Integration setup. To
complete them all, it should take around 10 min. The following tests are
launched, in the following order:

- **command line interface** test (`test_cli.py`): it verifies main arguments on
  the CLI interface. 
- **generate** test (`test_generate.py`): it creates trivial and random
  datasets based on 4 preprocessed MRIs obtained from OASIS dataset (testing
  dataset). The latter one can be [downloaded
  here](https://aramislab.paris.inria.fr/files/data/databases/tuto/OasisCaps2.tar.gz)
  and uncompressed into the `.clinicadl/test/data/dataset/` folder.
- **classify** test (`test_classify.py`): this test classifies synthetic
  (random) MRI obtained in previous test. You can preprocess the dataset
  obtained during the **generate** test (`clinica run deeplearning-prepare-data
  ./dataset/random-example image`) or you can [download it
  here](https://aramislab.paris.inria.fr/files/data/databases/tuto/RandomCaps.tar.gz).
  This test verifies that the output file exists. ([the previoulsy trained
  models are available
  here](https://aramislab.paris.inria.fr/files/data/models/dl/models_v002/)).
- **train** test (`test_train.py`): it runs training over the synthetic dataset
  and verifies that output folder structure was created. It needs to [download
  and uncompress this
  file](https://aramislab.paris.inria.fr/files/data/databases/tuto/labels_list.tar.gz)
  into the `.clinicadl/test/data/dataset/` folder and also the **RandomCaps**
  dataset downloaded in previous item.
- Several **tsvtool** functionalities (`test_tsvtool.py`). This test needs no
  data download as it is provided in the repo (`clinicadl/tests/data/tsvtool`).
  It test checks that:
    - the same label lists are found on an anonymized version of ADNI
      (`getlabels`),
    - data splits do not lead to data leakage and can correctly be found by
      loading functions (`split` and `kfold`),
    - the analysis tool runs and gives the same result as before on an
      anonimyzed version of ADNI (`analysis`).

To run each of these tests, a folder called `.clinicadl/test/data/` contains
the files used during the test execution. As mentioned above, some tests need
to download extra data. Testing datasets must be extracted inside a folder
named `.clinicadl/test/data/dataset/`. Trained models must be uncompresed
inside a folder called `.clinicadl/test/data/models/`.

Finally, be sure to have installed  the `pytest` library in order to run the
test suite (it's not a requirement of the main package).  Once everything is on
place, each of these tests can be run using the following command:

```
pytest  --verbose test_cli.py
```

to launch the _command line interface_ test. A similar command is used to
launch the other tests. If you don't run them in order, be sure of downloading
the necessary artifacts for the test.
