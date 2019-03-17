Node: save_as_pt (utility)
==========================

 Hierarchy : t1w_postprocessing_dl.save_as_pt
 Exec ID : save_as_pt

Original Inputs
---------------

* function_str : def save_as_pt(input_img):
    """
    This function is to transfer nii.gz file into .pt format, in order to train the classifiers model more efficient when loading the data.
    :param input_img:
    :return:
    """

    import torch, os
    import nibabel as nib

    image_array = nib.load(input_img).get_fdata()
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
    ## make sure the tensor dtype is torch.float32
    output_file = os.path.join(os.path.dirname(input_img), input_img.split('.nii.gz')[0] + '.pt')
    # save
    torch.save(image_tensor, output_file)

    return output_file

* ignore_exception : False
* input_img : ['/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_CAPS_test/subjects/sub-ADNI002S0619/ses-M00/t1/preprocessing_dl/sub-ADNI002S0619_ses-M00_space-MNI_res-1x1x1.nii.gz']

Execution Inputs
----------------

* function_str : def save_as_pt(input_img):
    """
    This function is to transfer nii.gz file into .pt format, in order to train the classifiers model more efficient when loading the data.
    :param input_img:
    :return:
    """

    import torch, os
    import nibabel as nib

    image_array = nib.load(input_img).get_fdata()
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
    ## make sure the tensor dtype is torch.float32
    output_file = os.path.join(os.path.dirname(input_img), input_img.split('.nii.gz')[0] + '.pt')
    # save
    torch.save(image_tensor, output_file)

    return output_file

* ignore_exception : False
* input_img : ['/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_CAPS_test/subjects/sub-ADNI002S0619/ses-M00/t1/preprocessing_dl/sub-ADNI002S0619_ses-M00_space-MNI_res-1x1x1.nii.gz']

Execution Outputs
-----------------

* output_file : ['/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_CAPS_test/subjects/sub-ADNI002S0619/ses-M00/t1/preprocessing_dl/sub-ADNI002S0619_ses-M00_space-MNI_res-1x1x1.pt']

Subnode reports
---------------

 subnode 0 : /teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Code/image_preprocessing/~/test/t1w_postprocessing_dl/save_as_pt/mapflow/_save_as_pt0/_report/report.rst

