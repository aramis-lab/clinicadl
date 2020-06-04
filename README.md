<h1 align="center">
  Clinica Deep Learning (<i>clinicadl</i>)
</h1>

<p align="center"><strong>for Alzheimer's Disease</strong></p>

<p align="center">
  <a href="https://ci.inria.fr/clinicadl/job/AD-DL/job/master/">
    <img src="https://ci.inria.fr/clinicadl/buildStatus/icon?job=AD-DL%2Fmaster" alt="Build Status">
  </a>
</p>

<p align="center">
  See also:
  <a href="#related-repositories">AD-ML</a>,
  <a href="#related-repositories">Clinica</a>
</p>


## About the project

This repository hosts the code source for reproducible experiments on
**automatic classification of Alzheimer's disease (AD) using anatomical MRI
data**.
It allows to train convolutional neural networks (CNN) models.
The journal version of the paper describing this work is available
[here](https://doi.org/10.1016/j.media.2020.101694).

Automatic classification of AD using a classical machine learning approach can
be performed using the software available here:
<https://github.com/aramis-lab/AD-ML>.

This software is currently in **active development**. If you find a problem
when using it, please [open an
issue](https://github.com/aramis-lab/ad-dl/issues).

Pretrained models for the CNN networks can be obtained here:
<https://zenodo.org/record/3491003>  

## Bibliography
All the papers described in the State of the art section of the manuscript may
be found at this URL address: <https://www.zotero.org/groups/2337160/ad-dl>.


## Main dependencies
- Python >= 3.6
- [Clinica](http://www.clinica.run/) >= 0.3.4 and [ANTs](https://stnava.github.io/ANTs/) (needs only to perform preprocessing)
- Numpy
- Pandas
- Scikit-learn
- Pytorch => 1.1
- Nilearn >= 0.5.3
- Nipy
- TensorBoardX

## How to install `clinicadl` ?

### Create a conda environment with the corresponding dependencies:
Keep the following order of the installation instructions.
It guaranties the right management of libraries depending on common packages:


```
conda create --name clinicadl_env python=3.6 pytorch torchvision -c pytorch

conda activate clinicadl_env
git clone git@github.com:aramis-lab/AD-DL.git
cd AD-DL
pip install -r requirements.txt
```

### Install the package `clinicadl` as developer in the active conda environment:

```
cd clinicadl
pip install -e .
```

## How to use `clinicadl` ?

`clinicadl` is an utility to be used with the command line.

To have an overview of the general options proposed by the software type:

```text
clinicadl -h

usage: clinicadl [-h] [--verbose]
{generate,preprocessing,extract,train,classify} ...

Clinica Deep Learning.

optional arguments:
-h, --help            show this help message and exit
--verbose, -v

Task to execute with clinicadl:
  What kind of task do you want to use with clinicadl? (preprocessing,
  extract, generate, train, validate, classify).

    {generate,tsvtool,preprocessing,extract,train,classify}
                        Tasks proposed by clinicadl
    tsvtool             Handle tsv files for metadata processing and data splits.
    generate            Generate synthetic data for functional tests.
    preprocessing       Prepare data for training (needs clinica installed).
    extract             Create data (slices or patches) for training.
    train               Train with your data and create a model.
    classify            Classify one image or a list of images with your
                        previously trained model.
```

### Tasks that can be performed by `clinicadl`

There are six kind of tasks that can be performed using the command line:

- **Process tsv files**. ``tsvtool`` includes many functions to get labels from BIDS, 
perform k-fold or single splits, produce demographic analysis of extracted labels
and reproduce the restrictions made on AIBL and OASIS in the original paper.

- **Generate a synthetic dataset.** Useful to run functional tests.

- **T1 MRI preprocessing.** It processes a dataset of T1 images stored in BIDS
  format and prepares to extract the tensors (see paper for details on the
  preprocessing). Output is stored using the
  [CAPS](http://www.clinica.run/doc/CAPS/Introduction/) hierarchy.

- **T1 MRI tensor extraction.** The `extract` option allows to create files in
  Pytorch format (`.pt`) with different options: the complete MRI, 2D slices
  and/or 3D patches. This files are also stored in the CAPS hierarchy.

- **Train neural networks.** Tensors obtained are used to perform the training of CNN models.

- **MRI classification.** Previously trained models can be used to perform the inference of a particular or a set of MRI.

For detailed instructions and options of each task type  `clinica 'task' -h`.

### Some examples

#### Labels extraction in tsv files

Typical use for `tsvtool getlabels`:

```text
clinicadl tsvtool getlabels <merged_tsv> <missing_mods> <results_path> --restriction_path <restriction_path>
```
where:

  - `<merged_tsv>` is the output file of `clinica iotools merge-tsv` command.
  - `<missing_mods>` is the folder containing the outputs of `clinica iotools missing-mods` command.
  - `<results_path>` is the path to the folder where tsv files are written.
  - `--restriction_path <restriction_path>` is a path to a tsv file containing the list of sessions that should be used.
  This argument is for example the result of a quality check procedure.
  
By default the extracted labels are only AD and CN, as OASIS database do not include
MCI patients. To include them add `--diagnoses AD CN MCI sMCI pMCI` at the end of the command.

#### Preprocessing
Typical use for `preprocessing` ([ANTs](https://stnava.github.io/ANTs/) software needs to be installed):

```text
clinicadl preprocessing <bids_dir> <caps_dir> <working_dir> --np 32
```
where:

  - `<bids_dir>` is the input folder containing the dataset in a [BIDS](http://www.clinica.run/doc/BIDS/) hierarchy.
  - `<caps_dir>` is the output folder containing the results in a [CAPS](http://www.clinica.run/doc/CAPS/Specifications/) hierarchy.
  - `<working_dir>` is the temporary directory to store pipelines intermediate results.
  - `--np <N>` (optional) is the number of cores used to run in parallel (in the example, 32 cores are used).

If you want to run the pipeline on a subset of your BIDS dataset, you can use
the `-tsv` flag to specify in a TSV file the participants belonging to your
subset.

#### Tensor extraction

These are the options available for the `extract` task:
```text
usage: clinicadl extract [-h] [-psz PATCH_SIZE] [-ssz STRIDE_SIZE]
                         [-sd SLICE_DIRECTION] [-sm {original,rgb}]
                         [-np NPROC]
                         caps_dir tsv_file working_dir {slice,patch,whole}

positional arguments:
  caps_dir              Data using CAPS structure.
  tsv_file              tsv file with subjects/sessions to process.
  working_dir           Working directory to save temporary file.
  {slice,patch,whole}   Method used to extract features. Three options:
                        'slice' to get 2D slices from the MRI, 'patch' to get
                        3D volumetric patches or 'whole' to get the complete
                        MRI.

optional arguments:
  -h, --help            show this help message and exit
  -psz PATCH_SIZE, --patch_size PATCH_SIZE
                        Patch size (only for 'patch' extraction) e.g:
                        --patch_size 50
  -ssz STRIDE_SIZE, --stride_size STRIDE_SIZE
                        Stride size (only for 'patch' extraction) e.g.:
                        --stride_size 50
  -sd SLICE_DIRECTION, --slice_direction SLICE_DIRECTION
                        Slice direction (only for 'slice' extraction). Three
                        options: '0' -> Sagittal plane, '1' -> Coronal plane
                        or '2' -> Axial plane
  -sm {original,rgb}, --slice_mode {original,rgb}
                        Slice mode (only for 'slice' extraction). Two options:
                        'original' to save one single channel (intensity),
                        'rgb' to saves three channel (with same intensity).
  -np NPROC, --nproc NPROC
                        Number of cores used for processing
```

## Run testing.

### Unit testing (WIP)

Be sure to have the `pytest` library in order to run the test suite.  This test
suite includes unit testing to be launched using the command line.

For the moment only the CLI (command line interface) part is tested using
`pytest`. We are planning to provide unit tests for the other tasks in the
future. If you want to run successfully the tests maybe you can use a command
like this one:
```text
pytest clinicadl/tests/test_cli.py
```

### Model prediction tests

For sanity check trivial datasets can be generated to train or test/validate
the predictive models.

The follow command allow you to generate two kinds of synthetic datasets: fully
separable (trivial) or intractable data (IRM with random noise added).
```text
python clinicadl generate {random,trivial} caps_directory tsv_path output_directory
```
The intractable dataset will be made of noisy versions of the first image of
the tsv file given at
`tsv_path` associated to random labels.

The trivial dataset includes two labels:
- AD corresponding to images with the left half of the brain with lower intensities,
- CN corresponding to images with the right half of the brain with lower intensities.

## Related Repositories

- [Clinica: Software platform for clinical neuroimaging studies](https://github.com/aramis-lab/clinica)
- [AD-ML: Framework for the reproducible classification of Alzheimer's disease using machine learning](https://github.com/aramis-lab/AD-ML)
