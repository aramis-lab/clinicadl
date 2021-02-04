# `t1-extensive` - Skull-stripping of T1-weighted MR images in MNI standard space

This pipeline performs skull-stripping of T1-weighted MR images in
[`Ixi549Space`
space](https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html)
using the brain mask (`mask_ICV.nii`) from the
[SPM](http://www.fil.ion.ucl.ac.uk/spm/) software.


## Prerequisites
You need to execute the [`t1-volume`
pipeline](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/T1_Volume/)
from Clinica to run this pipeline. Only the `t1-volume-tissue-segmentation`
sub-pipeline ("Tissue segmentation, bias correction and spatial normalization
to MNI space" step) needs to be executed.


### Running the pipeline
The pipeline can be run with the following command line:
```{.sourceCode .bash}
clinicadl preprocessing run t1-extensive <caps_directory>
```
where:

- `caps_directory` (str) is the input/output folder containing the results in a 
[CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.


!!! tip
    type `clinicadl preprocessing run t1-extensive --help` to see the full list of
    parameters.

### Outputs
Results are stored in the following folder of the
[CAPS hierarchy](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/):
`subjects/sub-<participant_label>/ses-<session_label>/t1_extensive` with the following outputs:

- `<source_file>_space-Ixi549Space_desc-SkullStripped_T1w.nii.gz`: T1w image
  non-linearly registered to the [`Ixi549Space`
  space](https://bids-specification.readthedocs.io/en/stable/99-appendices/08-coordinate-systems.html)
  and cropped.
