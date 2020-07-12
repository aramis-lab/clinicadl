# `classify` - Use previously trained models to perform the inference of a particular or a set of MRI

The tool `clinicadl classify` is used to perform the inference step using a
previously trained model on simple/multiple image.

<details>
<summary>
These are the options available for this task
</summary>

```{.sourceCode .bash}
usage: clinicadl classify [-h] [-cpu] caps_dir tsv_file model_path output_dir

positional arguments:
  caps_dir         Data using CAPS structure.
  tsv_file         TSV file with subjects/sessions to process.
  model_path       Path to the folder where the model and the json file are
                   stored.
  output_dir       Folder containing results of the training.

optional arguments:
  -h, --help       show this help message and exit
  -cpu, --use_cpu  Uses CPU instead of GPU.
```
</details>
