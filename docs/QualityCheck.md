# `quality_check` - Evaluate registration quality

The quality check procedure relies on a pretrained network that learned to classify images 
that are adequately registered to a template from others for which the registration failed. 
It reproduces the quality check procedure performed in [[Wen et al., 2020](https://doi.org/10.1016/j.media.2020.101694)]. 
It is an adaptation of [[Fonov et al., 2018](https://www.biorxiv.org/content/10.1101/303487v1)], using their pretrained models. 
Their original code can be found on [GitHub](https://github.com/vfonov/deep-qc).

!!! warning
    This quality check procedure is specific to the `t1-linear` pipeline and should not be applied 
    to other preprocessing procedures as the results may not be reliable.
    Moreover you should be aware that this procedure may not be well adapted to anonymized data 
    (for example images from OASIS-1) where parts of the images were removed or modified to guarantee anonymization.


## Prerequisites
You need to execute the `clinicadl preprocessing` and `clinicadl extract` pipelines prior to running this pipeline.

## Running the task
The pipeline can be run with the following command line:
```
clinicadl quality_check <caps_directory> <tsv_path> <output_path>
```
where:

- `caps_directory`(str) is the folder containing the results of the [`t1-linear` pipeline](./Run/T1_Linear.md) 
and the output of the present command, both in a [CAPS hierarchy](http://www.clinica.run/doc/CAPS/Introduction).
- `tsv_path` (str) is the path to a TSV file containing the subjects/sessions list to process (filename included).
- `output_path` (str) is the path to the output TSV file (filename included).


Pipeline options:

- `--threshold` (float) is the threshold applied to the output probability when deciding if the image passed or failed. 
Default value: `0.5`.
- `--batch_size` (int) is the size of the batch used in the DataLoader. Default value: `1`.
- `--nproc` (int) is the number of workers used by the DataLoader. Default value: `2`.
- `--use_cpu` (bool) forces to use CPU. Default behaviour is to try to use a GPU and to raise an error if it is not found.

## Outputs

The output of the pipeline is a TSV file in which all the sessions (identified with their `participant_id` and `session_id`) 
are associated with a `pass_probability` value and a True/False `pass` value depending on the chosen threshold. 
An example of TSV file is:

| **participant_id** | **session_id** | **pass_probability**   | **pass**  |
|--------------------|----------------|------------------------|-----------|
| sub-CLNC01         | ses-M00        | 0.9936990737915039     | True      |
| sub-CLNC02         | ses-M00        | 0.9772214889526367     | True      |
| sub-CLNC03         | ses-M00        | 0.7292165160179138     | True      |
| sub-CLNC04         | ses-M00        | 0.1549495905637741     | False     |
| ...                |  ...           |  ...                   |  ...      |
