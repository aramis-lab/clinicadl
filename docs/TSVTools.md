# `tsvtool` - Prepare your metadata

This collection of tools aims at handling metadata of BIDS-formatted datasets.
These tools perform three main tasks:

- Get the labels used in the classification task (`getlabels`),
- Split data to define test, validation and train cohorts (`split` + `kfold` or `prepare-experiment`),
- Analyze populations of interest (`analysis`).

!!! tip
    Classical ratios in the scientific literature are 80%-20% (or 70%-30%) for train/validation. 
    These values can be modified according to the size of the dataset, and the number of hyperparameters
    that are tuned.
    More information on the subject can be found [online](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7).


## `getlabels` - Extract labels specific to Alzheimer's disease

### Description

This tool writes a unique TSV file containing the labels asked by the user.
The file contains a column `group` which is the diagnosis of the subject for the session, and a column `subgroup` which shows if the diagnosis is stable or not.

The group correspond to the following description :

- **CN** (cognitively normal): sessions of subjects who were diagnosed as cognitively normal during all their follow-up;
- **AD** (Alzheimer's disease): sessions of subjects who were diagnosed as demented during all their follow-up;
- **MCI** (mild cognitive impairment): sessions of subjects who were diagnosed as prodromal (i.e. MCI) at baseline, 

These labels are specific to the Alzheimer's disease context and can only be extracted from cohorts used in [(Wen et al., 2020)](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300591).


The subgroup corresponds to the following description:
- **s** (stable): diagnosis remains identical during the `time_horizon` period following the current visit,
- **p** (progressive): diagnosis progresses to the following state during the `time_horizon` period following the current visit (eg. MCI --> AD),
- **r** (regressive): diagnosis regresses to the previous state during the `time_horizon` period following the current visit (eg. MCI --> CN),
- **uk** (unknown): there are not enough sessions to assess the reliability of the label but no changes were spotted,
- **us** (unstable): otherwise (multiple conversions / regressions).


However, users can define other label lists manually and give them in inputs of other functions of `tsvtools`. These TSV files will have to include the following columns: `participant_id`, `session_id`, and the name of the value used for the label (for example `diagnosis`).
Other functions of `tsvtool` may also try to have similar distributions according to the age and the sex of the participants. To benefit from this feature, the user must also include these two columns in their TSV files.

#### Example of TSV produced by `getlabels`:

| participant_id | session_id | group | subgroup | age | sex | ... |
| -- | -- | -- | -- | -- | -- | -- |
| sub-CLNC0001  | ses-M00 | MCI | pMCI | 72 | M | ... |
| sub-CLNC0001  | ses-M12 | MCI | pMCI | 65 | F | ... |
| sub-CLNC0001  | ses-M36 | AD | sAD | 66 | F | ... |
| sub-CLNC0002  | ses-M00 | AD | ukAD | 89 | F | ... |



### Running the task

```bash
clinicadl tsvtools getlabels [OPTIONS] BIDS_DIRECTORY RESULTS_DIRECTORY
```
where:

  - `BIDS_DIRECTORY` (Path) is the input folder containing the dataset in a BIDS hierarchy.
  - `RESULTS_DIRECTORY` (Path) is the path to the folder where output TSV files will be written.

Options:

  - `--modality` (str) is the modality for which the sessions are selected. 
  Sessions which do not include the modality will be excluded from the outputs.
  The name of the modality must correspond to a column of the TSV files in `missing_mods`.
  Default value: `t1w`.
  - `--diagnoses` (List[str]) is the list of the labels that will be extracted. Labels can be either a group or a subgroup. These labels must be chosen from the combination of group and subgroup seen above. All the sessions for which the diagnosis is not in the given list will be remove in the output TSV file.
  Default will be focus on the Alzheimer's disease and only process AD and CN labels.
  - `--time_horizon` (int) is the time horizon in months that is used to assess the stability of the MCI subjects.
  Default value: `36`.
  - `--restriction_tsv` (Path) is a path to a TSV file containing the list of sessions that should be used.
  This argument is useful to integrate the result of a quality check procedure. Default will not perform any restriction.
  - `--variables_of_interest` (List[str]) is a list of columns present in `MERGED_TSV` that will be included
  in the outputs.
  - `--keep_smc` (bool) if given the SMC participants are kept in the `CN.tsv` file.
  Default setting remove these participants.
  - `--caps_directory` (Path) is a folder containing a CAPS compliant dataset
  - `--merge_tsv` (Path) is a path to a TSV file containing the results of `clinica iotools merge-tsv` command. 
    - If not run before, this command will be run in the task and the output will be save in the `RESULTS_DIRECTORY` given with the name `merge.tsv`. 
    - If run before, the output has to be store with the name `merge.tsv`in the `RESULTS_DIRECTORY` or the path has to be given with this option. It avoids to re-run the `merge-tsv`command which can be very long (more than 30min).
    
    
 - `--missing_mods` (Path) is a path to a TSV file containing the results of `clinica iotools missing-modalities` command. 
    - If not run before, this command will be run in the task and the output directory will be save in the `RESULTS_DIRECTORY` given with the name `missing_mods`. 
    - If run before, the output directory has to be store with the name `missing_mods`in the `RESULTS_DIRECTORY` or the path has to be given with this option. It avoids to re-run the `missing-modalities`command which can be very long (more than 30min).
  

#### Example of how to run the task :

```bash
clinicadl tsvtools getlabels Data/BIDS Results/getlabels.tsv --diagnosis AD--diagnosis pMCI --time_horizon 12 
```

### Output tree

The command will output a TSV file and a JSON file and will store intermediate results from `clinica iotools merge-tsv` or `clinicadl tsvtool missing_mods` commands:
<pre>
└── &lt;getlabels&gt;
    ├── getlabels.tsv
    ├── getlabels.json
    ├── merge.tsv
    └── missing_mods
        └── ...
</pre>


## `split` - Single split observing similar age and sex distributions

### Description

This tool splits all the data in order to have the same sex, age and group distributions in both sets produced.
The similarity of the age and sex distributions is assessed by a T-test
and chi-square test, respectively.


### Running the task

```bash
clinicadl tsvtools split [OPTIONS] FORMATTED_DATA_TSV
```
where:

  - `FORMATTED_DATA_TSV` (Path) is the TSV file with the data that are going to be split 
  (output of `clinicadl tsvtools getlabels|split|kfold`).

Options:

  - `--subset_name` (str) is the name of the subset that is complementary to train.
 Default value: `validation`.
  - `--n_test` (float) gives the number of subjects that will be put in the set complementary to train:

    - If > 1, corresponds to the number of subjects to put in set with name `subset_name`.
    - If < 1, proportion of subjects to put in set with name `subset_name`.
    - If = 0, no training set is created and the whole dataset is considered as one set
        with name `subset_name`.

    Default value: `100`.
  - `--p_age_threshold` (float) is the threshold on the p-value used for the T-test on age distributions.
  Default value: `0.80`.
  - `--p_sex_threshold` (float) is the threshold on the p-value used for the chi2 test on sex distributions.
  Default value: `0.80`.
  - `--ignore_demographics` (bool) is a flag that disable the use of age, sex and group to balance the split.
  Default value: `False`
  - `--categorical_split_variable` (str) is the name of a categorical variable used for a stratified shuffle split (in addition to age and sex selection).
  Default value: `None`
  - `--test_tsv`(str) is the path to the test file in tsv format to avoid keeping the test data in the train/validation set.
  Default value: `None`.

### Output tree

The command will generate a new tsv file `subset_name.tsv` containing the keys (`participants_id`, `session_id`) included in the subset_name set. This file will be stored in the same directory as th `FORMATED_DATA_TSV`.
   
</pre>

The columns of the produced TSV files are `participant_id`, `session_id`.
TSV files ending with `_baseline.tsv` only include the baseline session of each subject (or
the session closest to baseline if the latter does not exist).

## `kfold` - K-fold split

### Description

This tool splits data to perform a k-fold cross-validation.

### Running the task

```bash
clinicadl tsvtools kfold [OPTIONS] FORMATTED_DATA_TSV
```
where `FORMATTED_DATA_TSV` (str) is the TSV file containing the data that are going to be split
(output of `clinicadl tsvtool getlabels|split|kfold`).

Options:

  - `--subset_name` (str) is the name of the subset that is complementary to train.
  Default value: `validation`.
  - `--n_splits` (int) is the value of k. If 0 is given, all subjects are considered as test subjects.
  Default value: `5`.
  - `--stratification` (str) is the name of the variable used to stratify the k-fold split.
  By default, the value is `None` which means there is no stratification.
  - `--test_tsv`(str) is the path to the test file in tsv format to avoid keeping the test data in the train/validation set.
  Default value: `None`.
  

### Output tree

The command will generate a new tsv file `subset_name.tsv` stored in the same directory as the `FORMATTED_DATA_TSV` file and containing the keys (`participants_id`, `session_id`).
For each key, it explicits which set it belongs to for each split according to the following structure (example for a 2-fold validation):

| participant_id | session_id | split_index | split_type |
| -- | -- | -- | -- |
| sub-CLNC0001  | ses-M00 | 0 | train |
| sub-CLNC0001  | ses-M00 | 1  | validation |
| sub-CLNC0002  | ses-M00 | 0 | train |
| sub-CLNC0002  | ses-M00 | 1  | validation |
| sub-CLNC0002  | ses-M06 | 0 | train |
| sub-CLNC0002  | ses-M06 | 1  | N/A |
| sub-CLNC0003  | ses-M00 | 0 | validation |
| sub-CLNC0003  | ses-M00 | 1 | train |


## `prepare-experiment`

### Description

This tool performs a single split to prepare testing data and then can perform either k-fold or single split to prepare validation data. It is an easy way to quicly prepare you data with basic options.

### Running the task

```bash
clinicadl tsvtools prepare-experiment [OPTIONS] FORMATTED_DATA_TSV
```
where:

  - `FORMATTED_DATA_TSV` (Path) is a TSV file output of `clinicadl tsvtool getlabels|split|kfold`.

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
    - If it is a SingleSplit: it is the number of subjects top put in validation set if it is a SingleSplit.
    - If it is a k-fold split: it is the number of folds in the k-folds split.
    - If =0, there is no training set and the whole dataset is considered as a validation set.


## `analysis`

### Description

This tool writes a TSV file that summarizes the demographics and clinical distributions of the
asked labels.
Continuous variables are described with statistics (mean, standard deviation, minimum and maximum),
whereas categorical values are grouped by categories.
The variables of interest are: age, sex, mini-mental state examination (MMSE) and global clinical dementia rating (CDR).

### Running the task

```bash
clinicadl tsvtools analysis [OPTIONS] MERGED_TSV FORMATTED_DATA_DIRECTORY RESULTS_DIRECTORY
```
where:

  - `MERGED_TSV` (Path) is the output file of the `clinica iotools merge-tsv` commands. If th `clinicadl tsvtools getlabels` command was run before, this file already exists and is stored in the output folder of this command.
  - `FORMATTED_DATA_DIRECTORY` (Path) is a folder containing one TSV file per label (output of `clinicadl tsvtool getlabels|split|kfold`).
  - `RESULTS_DIRECTORY` (Path) is the path to the TSV file that will be written (filename included).

Options:

  - `--diagnoses` (List[str]) is the list of the labels that will be extracted.
   These labels must be chosen from {AD,CN,MCI,sMCI,pMCI}. Default will only process AD and CN labels.
