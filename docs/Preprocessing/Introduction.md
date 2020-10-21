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


## Available pipelines

For the experiments of [[Wen et al., 2020](https://doi.org/10.1016/j.media.2020.101694)], two preprocessing pipelines were developed:

- [`t1-linear` pipeline](Run/T1_Linear) (called "Minimal" preprocessing in the paper) where bias field correction and an **affine** registration to the MNI standard space are performed with the [ANTs](http://stnava.github.io/ANTs/) software.

- [`t1-extensive` pipeline](Run/T1_Extensive) (called "Extensive" preprocessing in the paper) where bias field correction, **non linear** registration to the MNI standard space and skull stripping are performed with the [SPM](http://www.fil.ion.ucl.ac.uk/spm/) software.

After running a preprocessing pipeline with `clinicadl preprocessing run`, its outputs can be formatted
into [PyTorch tensor format](Extract.md) and the quality of the preprocessing can be [evaluated](QualityCheck.md)
to remove sessions for which the preprocessing procedure has crashed. 

!!! info "Which pipeline should I use?"
    In [[Wen et al., 2020](https://doi.org/10.1016/j.media.2020.101694)] we showed that the “Minimal” and “Extensive” procedures led to comparable classification accuracies in the context of Alzheimer's disease. Our advice would be to use the `t1-linear` pipeline for its simplicity.
