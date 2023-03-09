# `tsvtools` - Prepare your metadata

This collection of tools aims at handling metadata of BIDS-formatted datasets.
These tools perform three main tasks:

- Get the labels used in the classification task (`get-labels`),
- Get the progression of Alzheimer's disease (`get-progression`),
- Split data to define test, validation and train cohorts (`split` + `kfold` or `prepare-experiment`),
- Get additional data (`get-metadata`),
- Analyze populations of interest (`analysis`).

!!! tip
    Classical ratios in the scientific literature are 80%-20% (or 70%-30%) for train/validation. 
    These values can be modified according to the size of the dataset, and the number of hyperparameters
    that are tuned.
    More information on the subject can be found [online](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7).


## `get-labels` - Extract labels

### Description

This tool writes a unique TSV file containing the labels asked by the user.
The labels correspond to the following description and are stored in the column named diagnosis:

- **CN** (cognitively normal): sessions of subjects who were diagnosed as cognitively normal during all their follow-up;
- **AD** (Alzheimer's disease): sessions of subjects who were diagnosed as demented during all their follow-up;
- **MCI** (mild cognitive impairment): sessions of subjects who were diagnosed as prodromal (i.e. MCI) at baseline, 

These labels are specific to the Alzheimer's disease context and can only be extracted from cohorts used in [(Wen et al., 2020)](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300591).


However, users can define other label lists manually and give them in inputs of other functions of `tsvtools`. These TSV files will have to include the following columns: `participant_id`, `session_id`, and the name of the value used for the label (for example `diagnosis`).
Other functions of `tsvtools` may also try to have similar distributions according to the age and the sex of the participants. To benefit from this feature, the user must also include these two columns in their TSV files.

#### Example of TSV produced by `get-labels`:

| participant_id | session_id | diagnosis | age | sex | ... |
| -------------- | ---------- | --------- | --- | --- | --- |
| sub-CLNC0001  | ses-M00 | MCI | 72 | M | ... |
| sub-CLNC0001  | ses-M12 | MCI | 65 | F | ... |
| sub-CLNC0001  | ses-M36 | AD | 66 | F | ... |
| sub-CLNC0002  | ses-M00 | CN | 89 | F | ... |


### Running the task

```bash
clinicadl tsvtools get-labels [OPTIONS] BIDS_DIRECTORY OUTPUT_TSV
```
where:

  - `BIDS_DIRECTORY` (Path) is the input folder containing the dataset in a BIDS hierarchy.
  - `OUTPUT_TSV` (Path) is the path to the output TSV file (filename included).

Options:

  - `--modality` (str) is the modality for which the sessions are selected. 
  Sessions which do not include the modality will be excluded from the outputs.
  The name of the modality must correspond to a column of the TSV files in `missing_mods`.
  Default value: `t1w`.
  - `--diagnoses` (List[str]) is the list of the labels that will be extracted. Labels can be either a group or a subgroup. These labels must be chosen from the combination of group and subgroup seen above. All the sessions for which the diagnosis is not in the given list will be remove in the output TSV file.
  Default will be focus on the Alzheimer's disease and only process AD and CN labels.
  - `--restriction_tsv` (Path) is a path to a TSV file containing the list of sessions that should be used.
  This argument is useful to integrate the result of a quality check procedure. Default will not perform any restriction.
  - `--variables_of_interest` (List[str]) is a list of columns present in `MERGED_TSV` that will be included
  in the outputs.
  - `--keep_smc` (bool) if given the SMC participants are kept in the `CN.tsv` file.
  Default setting removes these participants.
  - `--caps_directory` (Path) is a folder containing a CAPS compliant dataset
  - `--merged_tsv` (Path) is a path to a TSV file containing the results of `clinica iotools merge-tsv` command. 
    - If not run before, this command will be run in the task and the output will be save in the `RESULTS_DIRECTORY` given with the name `merged.tsv`. 
    - If run before, the output has to be store with the name `merged.tsv`in the `RESULTS_DIRECTORY` or the path has to be given with this option. It avoids to re-run the `merge-tsv`command which can be very long (more than 30min).
  - `--missing_mods` (Path) is a path to a TSV file containing the results of `clinica iotools missing-modalities` command. 
    - If not run before, this command will be run in the task and the output directory will be save in the `RESULTS_DIRECTORY` given with the name `missing_mods`. 
    - If run before, the output directory has to be store with the name `missing_mods` in the `RESULTS_DIRECTORY` or the path has to be given with this option. It avoids to re-run the `clinica iotools check-missing-modalities` command which can be very long (more than 30min, depending on the hardware).
  - `--remove-unique-session` (bool) if given, participants having only one session will be removed.
  Default setting keeps these participants.
  

#### Example of how to run the task :

```bash
clinicadl tsvtools get-labels Data/BIDS --merged-tsv merged.tsv --diagnosis AD --diagnosis MCI 
```

This commandline will create a new TSV file (`Data/labels.tsv`) with a list of `AD` and `MCI` participants from `Data/Bids`. This command will run `clinica iotools check-missing-modalities` but used the given `merged.tsv` file.

### Output tree

The command will output a TSV file and a JSON file and will store intermediate results from `clinica iotools merge-tsv` or `clinicadl tsvtool missing_mods` commands:
<pre>
└── &lt;Results&gt;
    ├── labels.tsv
    ├── labels.json
    ├── merge.tsv
    └── missing_mods
        └── ...
</pre>


## `split` - Single split observing similar age and sex distributions

### Description

This tool splits all the data in order to have the same sex, age and diagnosis distributions in both sets produced.
The similarity of the age and sex distributions is assessed by a T-test
and chi-square test, respectively.


### Running the task

```bash
clinicadl tsvtools split [OPTIONS] DATA_TSV
```
where:

  - `DATA_TSV` (Path) is the TSV file with the data that are going to be split 
  (output of `clinicadl tsvtools get-labels|split|kfold`).

Options:

  - `--subset_name` (str) is the name of the subset that is complementary to train. Default value: `validation`.
  - `--n_test` (float) gives the number of subjects that will be put in the set complementary to train. Default value: `100`.

    - If > 1, corresponds to the number of subjects to put in set with name `subset_name`.
    - If < 1, proportion of subjects to put in set with name `subset_name`.
    - If = 0, no training set is created and the whole dataset is considered as one set
        with name `subset_name`.
    
  - `--p_age_threshold` (float) is the threshold on the p-value used for the T-test on age distributions.
  Default value: `0.80`.
  - `--p_sex_threshold` (float) is the threshold on the p-value used for the chi2 test on sex distributions.
  Default value: `0.80`.
  - `--ignore_demographics` (bool) is a flag that disable the use of age, sex and group to balance the split.
  Default value: `False`
  - `--categorical_split_variable` (str) is the name of a categorical variable used for a stratified shuffle split (in addition to age and sex selection).
  Default value: `None`

!!!tip
    Even if the split is performed on age and sex, these columns do not need to be part of the columns. If they are not present, the pipeline will try to go back to the labels.tsv to find them.
    
### Output tree

The command will generate new tsv files, one containing the keys included in the `subset_name` set and one containing the ones included in the `train` set. These files will be stored in the same directory as the `FORMATED_DATA_TSV`.

<pre>
└── &lt;split&gt;
    ├── train.tsv
    ├── train_baseline.tsv
    └── subset_name_baseline.tsv

</pre>


The columns of the produced TSV files are only the keys :`participants_id`, `session_id`.
TSV files ending with `_baseline.tsv` only include the baseline session of each subject (or
the session closest to baseline if the latter does not exist).

## `kfold` - K-fold split

### Description

This tool splits data to perform a k-fold cross-validation.

### Running the task

```bash
clinicadl tsvtools kfold [OPTIONS] DATA_TSV
```
where `DATA_TSV` (str) is the TSV file containing the data that are going to be split
(output of `clinicadl tsvtool getlabels|split|kfold`).

Options:

  - `--subset_name` (str) is the name of the subset that is complementary to train.
  Default value: `validation`.
  - `--n_splits` (int) is the value of k. If 0 is given, all subjects are considered as test subjects.
  Default value: `5`.
  - `--stratification` (str) is the name of the variable used to stratify the k-fold split.
  By default, the value is `None` which means there is no stratification.



### Output tree

The command will generate a new folder `k-fold` stored in the same directory as the `DATA_TSV` file and containing the different split folders. Each of these files contains the keys (`participants_id`, `session_id`).
For each key, it explicits which set it belongs to for each split according to the following structure (example for a 2-fold validation):
```
└── 2_fold
        ├── split_0
        │   ├── train_baseline.tsv
        │   ├── train.tsv
        │   └── validation_baseline.tsv
        └── split_1
            ├── train_baseline.tsv
            ├── train.tsv
            └── validation_baseline.tsv
    
```


## `prepare-experiment`

### Description

This tool performs a single split to prepare testing data and then can perform either k-fold or single split to prepare validation data. It is an easy way to quickly prepare data with basic options.

### Running the task

```bash
clinicadl tsvtools prepare-experiment [OPTIONS] DATA_TSV
```
where:

  - `DATA_TSV` (Path) is a TSV file output of `clinicadl tsvtool get-labels|split|kfold`.

Options:

  - `--n_test` (float) gives the number of subjects that will be put in the test set:

    - If > 1, corresponds to the number of subjects to put in set with name `subset_name`.
    - If < 1, proportion of subjects to put in set with name `subset_name`.
    - If = 0, no training set is created and the whole dataset is considered as one set
        with name `subset_name`.

    Default value: `100`.
  - `--validation_type` (str) is the name of the split wanted for the validation. It can only be `split`or `kfold`.
  Default value: `split`.
  - `--n_validation`(float) gives the number of subjects that will be put in the validation set:
    - For `split`, it is the number of subjects to put in validation set if it is a SingleSplit.
    - For `kfold`, it is the number of folds in the k-folds split.
    - If = 0, there is no training set and the whole dataset is considered as a validation set.

!!!tip
    Even if the split is performed on age and sex, these columns do not need to be part of the columns. If they are not present, the pipeline will go back to the labels.tsv to find them.

## `get-progression`

### Description

This tool adds a new column `progression` to the TSV file given. It corresponds to the progression of the Alzheimer's disease. The diagnosis must be one of those : `CN` (cognitively norma), `MCI` (mild cognitive impairment) or `AD` (alzheimer's disease).

The progression label corresponds to the following description:
- **s** (stable): diagnosis remains identical during the `time_horizon` period following the current visit,
- **p** (progressive): diagnosis progresses to the following state during the `time_horizon` period following the current visit (eg. MCI --> AD),
- **r** (regressive): diagnosis regresses to the previous state during the `time_horizon` period following the current visit (eg. MCI --> CN),
- **uk** (unknown): there are not enough sessions to assess the reliability of the label but no changes were spotted,
- **us** (unstable): otherwise (multiple conversions / regressions).
- 
### Running the task

```bash
clinicadl tsvtools get-progression [OPTIONS] DATA_TSV
```
where `DATA_TSV` (str) is the TSV file containing the data 
(output of `clinicadl tsvtool getlabels|split|kfold`).

Options:
  - `--time_horizon` (int) is the time horizon in months that is used to assess the stability of the MCI subjects.
  Default value: `36`.

!!!tip
    The diagnosis column do not need to be part of the columns, the pipeline will go back to the labels.tsv to calculate the progression.
    
## `get-metadata`

### Description

This tool add extra columns to the tsv file given. 
For example, the output of `clinicadl tsvtools split/kfold` returns a TSV file with only two columns `participant_id` and `session_id` and with this command, it is possible to retrieve the age, the diagnosis, the sex or other variables of interest in other tsv files.

### Running the task

```bash
clinicadl tsvtools get-metadata [OPTIONS] DATA_TSV MERGED_TSV
```
where:

  - `DATA_TSV` (Path) is a folder containing one TSV file per label (output of `clinicadl tsvtool get-labels|split|kfold`).
  - `MERGED_TSV` (Path) is a path to a TSV file containing the results of `clinica iotools merge-tsv` command.

Options:

  - `--variables_of_interest` (List[str]) is a list of columns present in `MERGED_TSV` that will be included in the outputs.

If no `variables_of_interest`is given, all columns present in `MERGED_TSV` will be included.

## `analysis`

### Description

This tool writes a TSV file that summarizes the demographics and clinical distributions of the asked labels. Continuous variables are described with statistics (mean, standard deviation, minimum and maximum), whereas categorical values are grouped by categories. The variables of interest are: age, sex, mini-mental state examination (MMSE) and global clinical dementia rating (CDR).

### Running the task

```bash
clinicadl tsvtools analysis [OPTIONS] MERGED_TSV DATA_TSV OUTPUT_TSV
```
where:

  - `MERGED_TSV` (Path) is the output file of the `clinica iotools merge-tsv` commands. If th `clinicadl tsvtools getlabels` command was run before, this file already exists and is stored in the output folder of this command.
  - `DATA_TSV` (Path) is a folder containing one TSV file per label (output of `clinicadl tsvtool getlabels|split|kfold`).
  - `OUTPUT_TSV` (Path) is the path to the TSV file that will be written (filename included).

Options:

  - `--diagnoses` (List[str]) is the list of the labels that will be extracted.
   These labels must be chosen from {AD,CN,MCI}. Default will only process AD and CN labels.
   

## Summary

This summary is a little help to better understand how to pre-process data before training a network, using [Clinica](https://aramislab.paris.inria.fr/clinica/docs/public/latest/) and ClinicaDL.

![tsvtools](https://user-images.githubusercontent.com/57992134/196130331-b93f123c-d607-45be-b3ad-a6e41cf0efce.png)




