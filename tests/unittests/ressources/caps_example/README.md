# Subject/session: 

### t1-linear (fake data): 
- sub-000: ses-M000 (**cropped** and **uncropped**), ses-M006
- sub-001: ses-M000 (**cropped** and **uncropped**, **missing tensor for uncropped**)
- sub-002: ses-M006 (**cropped** and **uncropped**)
- sub-004: ses-M000, ses-M006, ses-M0018
- sub-005: ses-M004, ses-M012

### pet-linear (real data, with hippocampus mask, suvr=pons2, trc=18FAV45): 
- sub-000: ses-M006 (**voxels=(1.2, 0.9, 0.9)**)
- sub-002: ses-M006 (**voxels=(0.9, 0.9, 0.9), .pt mask missing**), ses-M018 (**voxels=(0.9, -0.9, 0.9)**)

# Masks

- 1 common left hippocampus mask (**voxels=(0.9, 0.9, 0.9)**, **with .pt**)
- 1 common brain mask (**voxels=(0.9, 0.9, 0.9)**)
- 1 common right hemisphere mask (**voxels=(0.9, -0.9, 0.9)**)