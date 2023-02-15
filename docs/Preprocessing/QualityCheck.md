# `quality-check` - Evaluate registration quality

Two different quality check procedures are available in ClinicaDL:
one for the `t1-linear` preprocessing pipeline and another for the `t1-volume` 
pipeline.


## `quality-check t1-linear` - Evaluate `t1-linear` registration

The quality check procedure relies on a pretrained network that learned to classify images 
that are adequately registered to a template from others for which the registration failed. 
It reproduces the quality check procedure performed in [[Wen et al., 2020](https://doi.org/10.1016/j.media.2020.101694)]. 
It is an adaptation of [[Fonov et al., 2018](https://www.biorxiv.org/content/10.1101/303487v1)], using their pretrained models. 
Their original code can be found on [GitHub](https://github.com/vfonov/deep-qc).

!!! warning
    This quality check procedure is specific to the `t1-linear` pipeline and should not be applied 
    to other preprocessing procedures as the results may not be reliable.
    Moreover, you should be aware that this procedure may not be well adapted to de-identified data 
    (for example images from OASIS-1) where parts of the images were removed (e.g. the face)
    or modified to guarantee anonymization.


### Prerequisites
You need to execute the `clinica run t1-linear` and `clinicadl prepare-data` pipelines 
prior to running this task.

### Running the task
The task can be run with the following command line:
```
clinicadl quality-check t1-linear [OPTIONS] CAPS_DIRECTORY OUTPUT_TSV
```
where:

- `CAPS_DIRECTORY` (Path) is the folder containing the results of the [`t1-linear` pipeline](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/T1_Linear/) 
and the output of the present command, both in a [CAPS hierarchy](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
- `OUTPUT_TSV` (str) is the path to the output TSV file (filename included).


Options:

- `--participants_tsv` (Path) is the path to a TSV file containing the subjects/sessions list to check (filename included).
Default will process all sessions available in `caps_directory`.
- `--threshold` (float) is the threshold applied to the output probability when deciding if the image passed or failed. 
Default value: `0.5`.
- `--batch_size` (int) is the size of the batch used in the DataLoader. Default value: `1`.
- `--n_proc` (int) is the number of workers used by the DataLoader. Default value: `2`.
- `--gpu/--no_gpu` (bool) Use GPU for computing optimization. Default behaviour is to try to use a GPU and to raise an error if it is not found.
- `--use_tensor` (bool) is a flag allowing the pipeline to run on the extracted tensors and not on the nifti images. 
- `--network` (str) is the architecture chosen for the network (to chose between `darq`, `sq101` and `deep_qc`)

### Outputs

The output of the quality check is a TSV file in which all the sessions (identified with their `participant_id` and `session_id`) 
are associated with a `pass_probability` value and a True/False `pass` value depending on the chosen threshold. 
An example of TSV file is:

| **participant_id** | **session_id** | **pass_probability**   |
|--------------------|----------------|------------------------|
| sub-CLNC01         | ses-M00        | 0.9936990737915039     |
| sub-CLNC02         | ses-M00        | 0.9772214889526367     |
| sub-CLNC03         | ses-M00        | 0.7292165160179138     |
| sub-CLNC04         | ses-M00        | 0.1549495905637741     |
| ...                |  ...           |  ...                   |

## `quality-check t1-volume` - Evaluate `t1-volume` registration and gray matter segmentation

The quality check procedure is based on thresholds on different statistics that were empirically
linked to images of bad quality. Three steps are performed to remove images with the following characteristics:

1. a maximum value below 0.95,
2. a percentage of non-zero values below 15% or higher than 50%,
3. a similarity with the DARTEL template around the frontal lobe below 0.40. The similarity
corresponds to the normalized mutual information. This allows checking that the eyes are not
included in the brain volume. 
    
!!! warning
    This quality check procedure is specific to the `t1-volume` pipeline and should not be applied 
    to other preprocessing procedures as the results may not be reliable.


### Prerequisites
You need to execute the `clinica run t1-volume` pipeline prior to running this task.

### Running the task
The task can be run with the following command line:
```
clinicadl quality-check t1-volume [OPTIONS] CAPS_DIRECTORY OUTPUT_DIRECTORY GROUP_LABEL
```
where:

- `CAPS_DIRECTORY` (Path) is the folder containing the results of the [`t1-volume` pipeline](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/T1_Volume/) 
and the output of the present command, both in a [CAPS hierarchy](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
- `OUTPUT_DIRECTORY` (Path) is the path to an output directory in which TSV files will be created.
- `GROUP_LABEL` (str) is the identifier for the group of subjects used to create the DARTEL template.
You can check which groups are available in the `groups/` folder of your `caps_directory`.


### Outputs

This pipeline outputs 4 files:

- `QC_metrics.tsv` containing the three QC metrics for all the images,
- `pass_step-1.tsv` including only the images which passed the first step,
- `pass_step-2.tsv` including only the images which passed the two first steps,
- `pass_step-3.tsv` including only the images which passed all the three steps.

!!! note "Manual quality check"
    This quality check is really conservative and may keep some images that are not of good quality.
    You may want to check the last images kept at each step to assess if their quality is good enough 
    for your application.
 