!!! warning
    Data for testing must be manually downloaded (see below). Make sure you
    have installed pytest & Co (`pip install -r
    https://raw.githubusercontent.com/aramis-lab/clinicadl/dev/requirements-dev.txt`)
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
- **predict** test (`test_predict.py`): this test predicts synthetic
  (random) MRI obtained in previous test. You can preprocess the dataset
  obtained during the **generate** test (`clinica run deeplearning-prepare-data
  ./dataset/random_example t1-linear image`) or you can [download it
  here](https://aramislab.paris.inria.fr/files/data/databases/tuto/RandomCaps.tar.gz).
  This test verifies that the output files exist. <!--([the previoulsy trained
  models are available
  here](https://aramislab.paris.inria.fr/files/data/models/dl/models_v002/)).-->
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
