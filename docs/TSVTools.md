# `tsvtool` - Prepare your metadata

This collection of tools aims at handling metadata of BIDS-formatted datasets.
These tools perform three main tasks:

- Get the labels used in the classification task (`restrict` + `getlabels`),
- Split data to define test, validation and train cohorts (`split` + `kfold`),
- Analyze populations of interest (`analysis`).

## `restrict` - Reproduce restrictions on specific datasets.

### Description

In the [original paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300591), 
specific restrictions were applied to datasets used for testing:

- in **OASIS**, cognitively normal subjects who were younger than the youngest demented patient (62 years old) were removed,
- in **AIBL**, subjects whose age could not be retrieved (because it is missing for all their sessions) were removed.

### Running the task

```bash
clinicadl tsvtool restrict <dataset> <merged_tsv> <results_path>
```

where:

  - `dataset` (str) is the name of the dataset. Choices are `OASIS` or `AIBL`.
  - `merged_tsv` (str) is the output file of the `clinica iotools merge-tsv` command.
  - `results_path` (str) is the path to the output tsv file (filename included).
  This tsv file comprises the same columns as `merged_tsv`.

!!! tip
    Add your custom restrictions in `clinicadl/tools/tsv/restriction.py` to make
    your own experiments reproducible.

## `getlabels` - Extract labels specific to Alzheimer's disease

### Description

This tool writes a tsv file for each label asked by the user.
The labels correspond to the following description:

- CN (cognitively normal): sessions of subjects who were diagnosed as cognitively normal during all their follow-up;
- AD (Alzheimer's disease): sessions of subjects who were diagnosed as demented during all their follow-up;
- MCI (mild cognitive impairment): sessions of subjects who were diagnosed as prodromal (i.e. MCI) at baseline, 
who did not encounter multiple reversions and conversions and who did not convert back to the cognitively normal status;
- pMCI (progressive MCI): sessions of subjects who were diagnosed as prodromal at baseline, 
and progressed to dementia during the `time_horizon` period following the current visit;
- sMCI (stable MCI): sessions of subjects who were diagnosed as prodromal at baseline,
remained stable during the 36 months `time_horizon` period following the current visit and 
never progressed to dementia nor converted back to the cognitively normal status.

### Running the task

```bash
clinicadl tsvtool getlabels <merged_tsv> <missing_mods> <results_path>
```
where:

  - `merged_tsv` (str) is the output file of the `clinica iotools merge-tsv` or `clinicadl tsvtool restrict` commands.
  - `missing_mods` (str) is the folder containing the outputs of the `clinica iotools missing-mods` command.
  - `results_path` (str) is the path to the folder where output tsv files will be written.

Options:

  - `--modality` (str) Modality for which the sessions are selected. 
  Sessions which do not include the modality will be excluded from the outputs.
  The name of the modality must correspond to a column of the tsv files in `missing_mods`.
  Default value: `t1w`.
  - `--diagnoses` (list of str) is the list of the labels that will be extracted.
   These labels must be chosen from {AD,CN,MCI,sMCI,pMCI}. Default will only process AD and CN labels.
  - `--time_horizon` (int) is the time horizon in months that is used to assess the stability of the MCI subjects.
  Default value: `36`.
  - `--restriction_path` (str) is a path to a tsv file containing the list of sessions that should be used.
  This argument is useful to integrate the result of a quality check procedure. Default will not perform any restriction.
  - `--variables_of_interest` (list of str) is a list of columns present in `merged_tsv` that will be included
  in the outputs.
  - `--keep_smc` (bool) if given the SMC participants are kept in the `CN.tsv` file.
  Default setting remove these participants.

### Output tree

The command will output one tsv file per label:
<pre>
└── &lt;results_path&gt;
    ├── AD.tsv
    ├── CN.tsv
    ├── MCI.tsv
    ├── sMCI.tsv
    └── pMCI.tsv
</pre>

Each tsv file comprises the `participant_id` and `session_id` values of all the sessions that correspond to the label.
The values of the column `diagnosis` are equal to the label name.
The age and sex are also included in the TSV files. The names of these columns depend on the 
columns of `merged_tsv`.

## `split` - Single split observing similar age and sex distributions

### Description

This tool independently splits each label in order to have the same sex and age distributions
in both sets produced.
The similarity of the age and sex distributions is assessed by a T-test
and a chi-square test, respectively.

By default, there is a special treatment of the MCI set and its subsets (sMCI and pMCI) to avoid
data leakage. However, if there are too few patients, this can prevent finding a split
with similar demographics for these labels.

### Running the task

```bash
clinicadl tsvtool split <formatted_data_path>
```
where:

  - `formatted_data_path` (str) is the folder containing a tsv file per label which is going to be split 
  (output of `clinicadl tsvtool getlabels|split|kfold`).

Options:
<ul>
  <li> <code>--subset_name</code> (str) is the name of the subset that is complementary to train.
  Default value: <code>test</code>.</li>
  <li> <code>--n_test</code> (float) gives the number of subjects that will be put in the set complementary to train:
    <ul>
    <li>If > 1, corresponds to the number of subjects to put in set with name <code>subset_name</code>.</li>
    <li>If < 1, proportion of subjects to put in set with name <code>subset_name</code>.</li>
    <li>If = 0, no training set is created and the whole dataset is considered as one set</li>
    with name <code>subset_name</code>.
    </ul>
  Default value: <code>100</code>.</li>
  <li> <code>--MCI_sub_categories</code> (bool) is a flag that disables the special treatment of the MCI set and its subsets.
  This will allow sets with more similar age and sex distributions, but it will cause 
  data leakage for transfer learning tasks involving these sets. Default value: <code>False</code>.</li>
  <li> <code>--p_val_threshold</code> is the threshold on the p-value used for the T-test on age distributions.
  Default value: <code>0.80</code>.</li>
  <li> <code>--t_val_threshold</code> is the threshold on the t-value used for the chi2 test on sex distributions.
  Default value: <code>0.0642</code>.</li>
</ul>

### Output tree

The command will generate the following output tree:
<pre>
└── <b>formatted_data_path</b>
    ├── label-1.tsv
    ├── ...
    ├── label-n.tsv
    ├── <b>train</b>
    |   ├── label-1.tsv
    |   ├── label-1_baseline.tsv
    |   ├── ...
    |   ├── label-n.tsv
    |   └── label-n_baseline.tsv 
    └── <b>test</b>
        ├── label-1_baseline.tsv
        ├── ...
        └── label-n_baseline.tsv 
</pre>

The columns of the produced tsv files are `participant_id`, `session_id` and `diagnosis`.
TSV files ending with `_baseline.tsv` only include the baseline session of each subject (or
the session closest to baseline if the latter does not exist).

## `kfold` - K-fold split

### Description

This tool independently splits each label to perform a k-fold cross-validation.

### Running the task

```bash
clinicadl tsvtool kfold <formatted_data_path>
```
where `formatted_data_path` (str) is the folder containing a tsv file per label which is going to be split
(output of `clinicadl tsvtool getlabels|split|kfold`).

Options:

  - `--subset_name` (str) is the name of the subset that is complementary to train.
  Default value: `validation`.
  - `--n_splits` (int) Value of k. If 0 is given, all subjects are considered as test subjects.
  Default value: `5`.
  - `--MCI_sub_categories` (bool) is a flag that disables the special treatment of the MCI set and its subsets.
  This will cause data leakage for transfer learning tasks involving these sets. Default value: `False`.

### Output tree

The command will generate the following output tree:
<pre>
└── <b>formatted_data_path</b>
    ├── label-1.tsv
    ├── ...
    ├── label-n.tsv
    ├── <b>train_splits-&lt&lt;n_splits&gt</b>
    |   ├── <b>split-0</b>
    |   ├── ...
    |   └── <b>split-&lt;n_splits-1&gt;</b>
    |       ├── label-1.tsv
    |       ├── label-1_baseline.tsv
    |       ├── ...
    |       ├── label-n.tsv
    |       └── label-n_baseline.tsv
    └── <b>&lt;subset_name&gt;_splits-&lt;n_splits&gt;</b>
        ├── <b>split-0</b>
        ├── ...
        └── <b>split-&lt;n_splits-1&gt;</b>
            ├── label-1.tsv
            ├── label-1_baseline.tsv
            ├── ...
            ├── label-n.tsv
            └── label-n_baseline.tsv  
</pre>

The columns of the produced tsv files are `participant_id`, `session_id` and `diagnosis`.
TSV files ending with `_baseline.tsv` only include the baseline session of each subject (or
the session closest to baseline if the latter does not exist).

## `analysis`

### Description

This tool writes a tsv file that summarizes the demographics and clinical distributions of the
asked labels.
Continuous variables are described with statistics (mean, standard deviation, minimum and maximum),
whereas categorical values are grouped by categories.
The variables of interest are: age, sex, mini-mental state examination (MMSE) and global clinical dementia rating (CDR).

### Running the task

```bash
clinicadl tsvtool analysis <merged_tsv> <formatted_data_path> <results_path>
```
where:

  - `merged_tsv` (str) is the output file of the `clinica iotools merge-tsv` or `clinicadl tsvtool restrict` commands.
  - `formatted_data_path` (str) is a folder containing one TSV file per label (output of `clinicadl tsvtool getlabels|split|kfold`).
  - `results_path` (str) is the path to the tsv file that will be written (filename included).

Options:

  - `--diagnoses` (list of str) is the list of the labels that will be extracted.
   These labels must be chosen from {AD,CN,MCI,sMCI,pMCI}. Default will only process AD and CN labels.
  - `--baseline` (bool) is a flag to perform the analysis on `<label>_baseline.tsv` files
  instead of `<label>.tsv` files comprising all the sessions.
  Default: `False`.
