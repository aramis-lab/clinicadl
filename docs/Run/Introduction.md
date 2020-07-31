# `clinicadl preprocessing` - Preprocessing pipelines

## Introduction

A proper image preprocessing procedure is a fundamental step to ensure a good classification performance, 
especially in the domain of MRI or PET. Although CNNs have the potential to extract low-to-high level features from the raw images, 
the influence of image preprocessing remained to be clarified (in particular for in Alzheimer's Disease classification 
where datasets are relatively small).



## Description of preprocessing steps

In the context of Alzheimer's Disease classification, image preprocessing procedures included:

- **Bias field correction:** MR images can be corrupted by a low frequency and smooth signal caused by magnetic field inhomogeneities. 
This bias field induces variations in the intensity of the same tissue in different locations of the image, 
which deteriorates the performance of image analysis algorithms such as registration.
- **Skull stripping:** Extracranial tissues can be an obstacle for image analysis algorithms​. 
A large number of methods have been developed for brain extraction, also called skull stripping, and many are implemented in software tools.
- **Image registration:** Medical image registration consists of spatially aligning two or more images, 
either globally (rigid and affine registration) or locally (non-rigid registration), 
so that voxels in corresponding positions contain comparable information.



## Available pipelines

For the experiments of ([Wen et al., 2020](https://doi.org/10.1016/j.media.2020.101694)), two preprocessing pipelines were developed:

- [`t1-linear` pipeline](Run/T1_Linear) (called "Minimal" preprocessing in the paper) 
where an **affine** registration to the MNI standard space is performed with the [ANTs](http://stnava.github.io/ANTs/) software.

- [`t1-extensive` pipeline](Run/T1_Extensive) (called "Extensive" preprocessing in the paper) 
where **non linear** registration to the MNI standard space is performed with the [SPM](http://www.fil.ion.ucl.ac.uk/spm/) software.

!!! info "Which preprocessing is adapted to my network?"
    In ([Wen et al., 2020](https://doi.org/10.1016/j.media.2020.101694)), `t1-linear` showed similar results to 
    `t1-extensive`, although it is simpler.
