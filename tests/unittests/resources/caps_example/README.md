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
            * label: 1.0
        * tensors
    * ses-M003:
        * niftis:
            * uncropped only
            * affine: diag(1.3, 1.2, 1.1) 
            * shape: (1, 1, 1)
            * label: 1.0
        * tensors
* sub-010:
    * ses-M003:
        * niftis:
            * uncropped only
            * affine: diag(1.3, 1.2, 1.1) 
            * shape: (1, 1, 1)
            * label: 2.0
        * tensors
    * ses-M012
        * niftis:
            * uncropped only
            * affine: diag(1.3, 1.2, 1.1) 
            * shape: (1, 1, 1)
            * label: 2.0
        * tensors
* sub-100:
    * ses-M000:
        * niftis:
            * cropped and uncropped
            * affine: diag(1, 1, 1) 
            * shape: (1, 1, 1)
            * label: None
        * tensors
    * ses-M012:
        * niftis:
            * uncropped only
            * affine: diag(1, 1, 1) 
            * shape: (1, 1, 1)
            * label: None
        * tensors
* sub-999:
    * ses-M099:
        * niftis:
            * cropped and uncropped
            * affine: diag(1, 1, 1) 
            * shape: (1, 1, 1)
            * label: None
        * tensors
    * ses-M999:
        * niftis:
            * uncropped only
            * affine: diag(1, 1, 1) 
            * shape: (1, 1, 1)
            * label: None
        * tensors

## common masks
* leftHipppocampus:
    * niftis:
        * affine: diag(1.3, 1.2, 1.1) 
        * shape: (3, 3, 3)
    * tensors:
        * shape: (2, 2, 2)
* rightHemisphere:
    * niftis:
        * affine: diag(1, 1, 1) 
        * shape: (1, 1, 1)