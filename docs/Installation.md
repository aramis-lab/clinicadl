# Installation

You will find below the steps for installing ClinicaDL on Linux or Mac.
Please do not hesitate to contact us on the
[forum](https://groups.google.com/forum/#!forum/clinica-user) or
[GitHub](https://github.com/aramis-lab/clinicadl/issues)
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
conda create --name clinicadlEnv python=3.8
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
    Usage: clinicadl [OPTIONS] COMMAND [ARGS]...

      ClinicaDL command line.

      For more information please read the doc: https://clinicadl.readthedocs.io/en/latest/
      Source code is available on GitHub: https://github.com/aramis-lab/clinicaDL .

      Do not hesitate to create an issue to report a bug or suggest an improvement.

    Options:
      --version      Show the version and exit.
      -v, --verbose  Increase logging verbosity.  [x>=0]
      -h, --help     Show this message and exit.

    Commands:
      extract        Extract Pytorch tensors from nifti images.
      generate       Generation of synthetic dataset.
      interpret      Interpretation of trained models using saliency map method.
      predict        Infer the outputs of a trained model on a test set.
      quality-check  Performs quality check procedure for t1-linear or t1-volume...
      random-search  Hyperparameter exploration using random search.
      resume         Resume training job in specified maps.
      train          Train a deep learning model on your neuroimaging dataset.
      tsvtool        Manipulation of TSV files to prepare and manage input data.
    ```


### Deactivation of the ClinicaDL environment
At the end of your session, you can deactivate your Conda environment:
```{.sourceCode .bash}
conda deactivate
```
