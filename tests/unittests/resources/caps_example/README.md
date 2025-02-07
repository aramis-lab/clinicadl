## t1
* sub-000:
    * ses-M000:
        * niftis:
            * uncropped only
            * affine: diag(1.3, 1.2, 1.1) 
            * shape: (3, 3, 3)
            * label: "seg"
            * masks: "brain"
        * tensors:
            * shape: (2, 2, 2)
    * ses-M003:
        * niftis:
            * uncropped only
            * affine: diag(1.3, 1.2, 1.1) 
            * shape: (3, 3, 3)
            * label: "seg", with affine=diag(1, 1, 1, 1)
            * masks: "brain", with shape=(2, 2, 2)
* sub-010:
    * ses-M003:
        * niftis:
            * uncropped only
            * affine: diag(1.3, 1.2, 1.1) 
            * shape: (3, 3, 3)
            * label: "seg"
            * masks: "brain"
    * ses-M012
        * niftis:
            * uncropped only
            * affine: diag(1, 1, 1) 
            * shape: (1, 1, 1)
            * label: "seg"
            * masks: "brain"

## pet (trc=18FAV45, suvr=pons2)
* sub-000:
    * ses-M000:
        * niftis:
            * uncropped only
            * extension: .nii
            * affine: diag(1.3, 1.2, 1.1) 
            * shape: (1, 1, 1)
        * tensors
            * label: 1.0
    * ses-M003:
        * niftis:
            * uncropped only
            * affine: diag(1.3, 1.2, 1.1) 
            * shape: (1, 1, 1)
        * tensors
            * label: 1.0
* sub-010:
    * ses-M003:
        * niftis:
            * uncropped only
            * affine: diag(1.3, 1.2, 1.1) 
            * shape: (1, 1, 1)
        * tensors
            * label: 2.0
    * ses-M012
        * niftis:
            * uncropped only
            * affine: diag(1.3, 1.2, 1.1) 
            * shape: (1, 1, 1)
        * tensors
            * label: 2.0
* sub-100:
    * ses-M000:
        * niftis:
            * cropped and uncropped
            * affine: diag(1, 1, 1) 
            * shape: (1, 1, 1)
        * tensors
            * label: None
    * ses-M012:
        * niftis:
            * uncropped only
            * affine: diag(1, 1, 1) 
            * shape: (1, 1, 1)
        * tensors
            * label: None
* sub-999:
    * ses-M099:
        * niftis:
            * cropped and uncropped
            * affine: diag(1, 1, 1) 
            * shape: (1, 1, 1)
        * tensors
            * label: None
    * ses-M999:
        * niftis:
            * uncropped only
            * affine: diag(1, 1, 1) 
            * shape: (1, 1, 1)
        * tensors
            * label: None

## common masks
* leftHipppocampus:
    * nifti:
        * affine: diag(1.3, 1.2, 1.1) 
        * shape: (3, 3, 3)
    * tensors:
        * shape: (2, 2, 2)
* rightHipppocampus:
    * nifti:
        * affine: diag(1, 1, 1) 
        * shape: (3, 3, 3)
* rightHemisphere:
    * nifti:
        * affine: diag(1, 1, 1) 
        * shape: (1, 1, 1)
        * .nii extension

## problematic participant, session
* sub-666:
    * ses-M666:
        * flair niftis:
            * uncropped only
            * .nii AND .nii.gz extensions