# Installation

You will find below the steps for installing `clinicadl` on Linux or Mac.
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
                     {generate,preprocessing,extract,train,predict,tsvtool} ...

    Deep learning software for neuroimaging datasets
    
    optional arguments:
      -h, --help            show this help message and exit
      -l file.log, --logname file.log
                            Define the log file name (default: clinicaDL.log)
      -V, --version         ClinicaDL's installed version
    
    Task to execute with clinicadl:
      What kind of task do you want to use with clinicadl?
    
      {generate,preprocessing,random-search,train,predict,tsvtool,interpret}
                            ****** Tasks proposed by clinicadl ******
        generate            Generate synthetic data for functional tests.
        preprocessing       Preprocess T1w-weighted images with t1-linear or
                            t1-extensive pipelines.
        random-search       Generate random networks to explore hyper parameters
                            space.
        train               Train with your data and create a model.
        predict             Performs the individual predictions of a list of
                            subject in tsv_path. If labels are given, will also
                            compute global metrics on the data set.
        tsvtool             Handle tsv files for metadata processing and data
                            splits.
        interpret           Interpret the prediction of a CNN with saliency maps.
    ```


### Deactivation of the ClinicaDL environment
At the end of your session, you can deactivate your Conda environment:
```{.sourceCode .bash}
conda deactivate
```
