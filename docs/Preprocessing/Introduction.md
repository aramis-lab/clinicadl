# `clinicadl preprocessing` - Image preprocessing pipelines

## Introduction

A proper image preprocessing procedure is a fundamental step to ensure a good classification performance, 
especially in the domain of MRI or PET. 
Although CNNs have the potential to extract low-to-high level features from the raw images, 
the influence of image preprocessing remains to be clarified (in particular for in Alzheimer's Disease 
classification where datasets are relatively small).

## Description of the main preprocessing steps for MR images

In the context of brain disease classification, MR image preprocessing procedures may include:

- **Bias field correction:** MR images can be corrupted by a low frequency and smooth signal caused by magnetic field inhomogeneities. 
This bias field induces variations in the intensity of the same tissue in different locations of the image, which deteriorates the performance
 of image analysis algorithms such as registration.
- **Skull stripping:** Extracranial tissues can be an obstacle for image analysis algorithms. 
A large number of methods have been developed for brain extraction, also called skull stripping, 
and many are implemented in software tools.
- **Image registration:** Medical image registration consists in spatially aligning two or more images, 
either globally (rigid and affine registration) or locally (non-rigid registration), 
so that voxels in corresponding positions contain comparable information.

Concerning PET image, preprocessing include:

- **Image registration:** Medical image registration consists in spatially aligning two or more images, 
either globally (rigid and affine registration) or locally (non-rigid registration), 
so that voxels in corresponding positions contain comparable information.
- **Intensity normalization:** using the average PET uptake in a reference region resulting in a standardized uptake value ratio (SUVR) map. Indeed PET images intensities greatly vary depending on the patient's anatomy and physiology, and the quantity of tracer injected. It is necessary to perform this intensity normalization to enable inter-subject comparisons.


## Clinica's pipelines

For the preprocessing of neuroimages, we encourage you to use the pipelines available in [Clinica](https://aramislab.paris.inria.fr/clinica/docs/public/latest/):

- [`t1-linear` pipeline](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/T1_Linear/) for T1w MRI where bias field correction and an affine registration to the MNI standard space are performed with the [ANTs](http://stnava.github.io/ANTs/) software.

- [`pet-linear` pipeline](https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/PET_Linear/) for PET scans where spatial normalization to the MNI space and intensity normalization are performed with the [ANTs](http://stnava.github.io/ANTs/) software.

After running a preprocessing pipeline with `clinica run`, its outputs can be formatted
into [PyTorch tensor format](Extract.md) and the quality of the preprocessing can be [evaluated](QualityCheck.md)
to remove sessions for which the preprocessing procedure has crashed. 
