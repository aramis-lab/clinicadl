Node: extract_slices (utility)
==============================

 Hierarchy : t1w_postprocessing_dl.extract_slices
 Exec ID : extract_slices

Original Inputs
---------------

* function_str : def extract_slices(preprocessed_T1, slice_direction=0, slice_mode='original'):
    """
    This is to extract the slices from three directions
    :param preprocessed_T1:
    :param slice_direction: which axis direction that the slices were extracted
    :return:
    """
    import torch, os

    image_tensor = torch.load(preprocessed_T1)
    ## reshape the tensor, delete the first dimension for slice-level
    image_tensor = image_tensor.view(image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3])

    ## sagital
    slice_list_sag = range(20, image_tensor.shape[0] - 20) # delete the first 20 slice and last 20 slices

    if slice_direction == 0:
        for index_slice in slice_list_sag:
            # for i in slice_list:
            ## sagital
            slice_select_sag = image_tensor[index_slice, :, :]

            ## convert the slices to images based on if transfer learning or not
            # train from scratch
            extracted_slice_original_sag = slice_select_sag.unsqueeze(0) ## shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_sag = (slice_select_sag - slice_select_sag.min()) / (slice_select_sag.max() - slice_select_sag.min())
            extracted_slice_rgb_sag = torch.stack((slice_select_sag, slice_select_sag, slice_select_sag)) ## shape should be 3 * W * L

            # save into .pt format
            if slice_mode == 'original':
                output_file_original = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-sag_originalslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_original_sag, output_file_original)
            elif slice_mode == 'rgb':
                output_file_rgb = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-sag_rgbslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_rgb_sag, output_file_rgb)

    elif slice_direction == 1:
        ## cornal
        slice_list_cor = range(15, image_tensor.shape[1] - 15) # delete the first 20 slice and last 15 slices
        for index_slice in slice_list_cor:
            # for i in slice_list:
            ## sagital
            slice_select_cor = image_tensor[:, index_slice, :]

            ## convert the slices to images based on if transfer learning or not
            # train from scratch
            extracted_slice_original_cor = slice_select_cor.unsqueeze(0) ## shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_cor = (slice_select_cor - slice_select_cor.min()) / (slice_select_cor.max() - slice_select_cor.min())
            extracted_slice_rgb_cor = torch.stack((slice_select_cor, slice_select_cor, slice_select_cor)) ## shape should be 3 * W * L

            # save into .pt format
            if slice_mode == 'original':
                output_file_original = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-cor_originalslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_original_cor, output_file_original)
            elif slice_mode == 'rgb':
                output_file_rgb = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-cor_rgblslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_rgb_cor, output_file_rgb)

    else:

        ## axial
        slice_list_axi = range(15, image_tensor.shape[2] - 15) # delete the first 20 slice and last 15 slices
        for index_slice in slice_list_axi:
            # for i in slice_list:
            ## sagital
            slice_select_axi = image_tensor[:, :, index_slice]

            ## convert the slices to images based on if transfer learning or not
            # train from scratch
            extracted_slice_original_axi = slice_select_axi.unsqueeze(0) ## shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_axi = (slice_select_axi - slice_select_axi.min()) / (slice_select_axi.max() - slice_select_axi.min())
            extracted_slice_rgb_axi = torch.stack((slice_select_axi, slice_select_axi, slice_select_axi)) ## shape should be 3 * W * L

            # save into .pt format
            if slice_mode == 'original':
                output_file_original = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-axi_originalslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_original_axi, output_file_original)
            elif slice_mode == 'rgb':
                output_file_rgb = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-axi_rgblslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_rgb_axi, output_file_rgb)

    return preprocessed_T1

* ignore_exception : False
* preprocessed_T1 : ['/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_CAPS_test/subjects/sub-ADNI002S0619/ses-M00/t1/preprocessing_dl/sub-ADNI002S0619_ses-M00_space-MNI_res-1x1x1.pt']
* slice_direction : 0
* slice_mode : original

Execution Inputs
----------------

* function_str : def extract_slices(preprocessed_T1, slice_direction=0, slice_mode='original'):
    """
    This is to extract the slices from three directions
    :param preprocessed_T1:
    :param slice_direction: which axis direction that the slices were extracted
    :return:
    """
    import torch, os

    image_tensor = torch.load(preprocessed_T1)
    ## reshape the tensor, delete the first dimension for slice-level
    image_tensor = image_tensor.view(image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3])

    ## sagital
    slice_list_sag = range(20, image_tensor.shape[0] - 20) # delete the first 20 slice and last 20 slices

    if slice_direction == 0:
        for index_slice in slice_list_sag:
            # for i in slice_list:
            ## sagital
            slice_select_sag = image_tensor[index_slice, :, :]

            ## convert the slices to images based on if transfer learning or not
            # train from scratch
            extracted_slice_original_sag = slice_select_sag.unsqueeze(0) ## shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_sag = (slice_select_sag - slice_select_sag.min()) / (slice_select_sag.max() - slice_select_sag.min())
            extracted_slice_rgb_sag = torch.stack((slice_select_sag, slice_select_sag, slice_select_sag)) ## shape should be 3 * W * L

            # save into .pt format
            if slice_mode == 'original':
                output_file_original = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-sag_originalslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_original_sag, output_file_original)
            elif slice_mode == 'rgb':
                output_file_rgb = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-sag_rgbslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_rgb_sag, output_file_rgb)

    elif slice_direction == 1:
        ## cornal
        slice_list_cor = range(15, image_tensor.shape[1] - 15) # delete the first 20 slice and last 15 slices
        for index_slice in slice_list_cor:
            # for i in slice_list:
            ## sagital
            slice_select_cor = image_tensor[:, index_slice, :]

            ## convert the slices to images based on if transfer learning or not
            # train from scratch
            extracted_slice_original_cor = slice_select_cor.unsqueeze(0) ## shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_cor = (slice_select_cor - slice_select_cor.min()) / (slice_select_cor.max() - slice_select_cor.min())
            extracted_slice_rgb_cor = torch.stack((slice_select_cor, slice_select_cor, slice_select_cor)) ## shape should be 3 * W * L

            # save into .pt format
            if slice_mode == 'original':
                output_file_original = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-cor_originalslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_original_cor, output_file_original)
            elif slice_mode == 'rgb':
                output_file_rgb = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-cor_rgblslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_rgb_cor, output_file_rgb)

    else:

        ## axial
        slice_list_axi = range(15, image_tensor.shape[2] - 15) # delete the first 20 slice and last 15 slices
        for index_slice in slice_list_axi:
            # for i in slice_list:
            ## sagital
            slice_select_axi = image_tensor[:, :, index_slice]

            ## convert the slices to images based on if transfer learning or not
            # train from scratch
            extracted_slice_original_axi = slice_select_axi.unsqueeze(0) ## shape should be 1 * W * L

            # train for transfer learning, creating the fake RGB image.
            slice_select_axi = (slice_select_axi - slice_select_axi.min()) / (slice_select_axi.max() - slice_select_axi.min())
            extracted_slice_rgb_axi = torch.stack((slice_select_axi, slice_select_axi, slice_select_axi)) ## shape should be 3 * W * L

            # save into .pt format
            if slice_mode == 'original':
                output_file_original = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-axi_originalslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_original_axi, output_file_original)
            elif slice_mode == 'rgb':
                output_file_rgb = os.path.join(os.path.dirname(preprocessed_T1), preprocessed_T1.split('.pt')[0] + '_axis-axi_rgblslice-' + str(index_slice) + '.pt')
                torch.save(extracted_slice_rgb_axi, output_file_rgb)

    return preprocessed_T1

* ignore_exception : False
* preprocessed_T1 : ['/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_CAPS_test/subjects/sub-ADNI002S0619/ses-M00/t1/preprocessing_dl/sub-ADNI002S0619_ses-M00_space-MNI_res-1x1x1.pt']
* slice_direction : 0
* slice_mode : original

Execution Outputs
-----------------

* preprocessed_T1 : ['/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_CAPS_test/subjects/sub-ADNI002S0619/ses-M00/t1/preprocessing_dl/sub-ADNI002S0619_ses-M00_space-MNI_res-1x1x1.pt']

Subnode reports
---------------

 subnode 0 : /teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Code/image_preprocessing/~/test/t1w_postprocessing_dl/extract_slices/mapflow/_extract_slices0/_report/report.rst

