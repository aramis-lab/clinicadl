# `tsvtool` - Extract label in TSV files

## `getlabels`

Typical use for `tsvtool getlabels`:

```{.sourceCode .bash}
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


<details>
<summary>
The full list of options available to obtain labels from tsv files.
</summary>

```{.sourceCode .bash}
usage: clinicadl tsvtool getlabels [-h] [--modality MODALITY]
                                   [--diagnoses {AD,CN,MCI,sMCI,pMCI} [{AD,CN,MCI,sMCI,pMCI} ...]]
                                   [--time_horizon TIME_HORIZON]
                                   [--restriction_path RESTRICTION_PATH]
                                   merged_tsv missing_mods results_path

positional arguments:
  merged_tsv            Path to the file obtained by the command clinica
                        iotools merge-tsv.
  missing_mods          Path to the folder where the outputs of clinica
                        iotools missing-mods are.
  results_path          Path to the folder where tsv files are extracted.

optional arguments:
  -h, --help            show this help message and exit
  --modality MODALITY, -mod MODALITY
                        Modality to select sessions. Sessions which do not
                        include the modality will be excluded.
  --diagnoses {AD,CN,MCI,sMCI,pMCI} [{AD,CN,MCI,sMCI,pMCI} ...]
                        Labels that must be extracted from merged_tsv.
  --time_horizon TIME_HORIZON
                        Time horizon to analyse stability of MCI subjects.
  --restriction_path RESTRICTION_PATH
                        Path to a tsv containing the sessions that can be
                        included.
```
</details>
