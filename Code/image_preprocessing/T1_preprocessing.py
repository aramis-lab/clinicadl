# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 14:17:41 2017

@author: Junhao WEN
"""

def preprocessing_t1w(bids_directory, caps_directory, tsv, ref_template, working_directory=None):
    """
    This is a preprocessing pipeline including three steps:
        1) N4 bias correction (Ants)
        2) linear registration to MNI (MNI icbm152 nlinear sym template) (ANTS)
        3) cropping the background to save the computational power
    Args:

    Returns:

    """

    from nipype.interfaces import ants
    import nipype.interfaces.io as nio
    import nipype.interfaces.utility as nutil
    import nipype.pipeline.engine as npe
    import tempfile
    from T1_preprocessing_utils import get_subid_sesid_datasink, bids_datagrabber_t1w, crop_nifti, rename_file, ants_registration_syn_quick, ants_histogram_intensity_normalization
    from clinica.pipelines.engine import get_subject_session_list

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    inputnode = npe.Node(nutil.IdentityInterface(
        fields=['bids_directory', 'tsv']),
        name='inputnode')
    inputnode.inputs.bids_directory = bids_directory
    inputnode.inputs.tsv = tsv

    get_subject_session_list = npe.Node(name='get_subject_session_list',
                               interface=nutil.Function(
                                   function=get_subject_session_list,
                                   input_names=['input_dir', 'ss_file', 'is_bids_dir'],
                                   output_names=['sessions', 'subjects']))
    get_subject_session_list.inputs.is_bids_dir = True

    # use pybids to grab the anat_t1 images based on the tsv file
    datagrabbernode = npe.Node(name='datagrabbernode',
                               interface=nutil.Function(
                                   function=bids_datagrabber_t1w,
                                   input_names=['input_dir', 'subject_list', 'session_list'],
                                   output_names=['anat_t1']))


    ## convert nii.gz to .mgz format
    n4biascorrection = npe.MapNode(name='n4biascorrection',
                                   iterfield=['input_image', 'output_image'],
                                   interface=ants.N4BiasFieldCorrection(
        dimension=3, save_bias=True, bspline_fitting_distance=600))

    antsRegistrationSyNQuick = npe.MapNode(name='antsRegistrationSyNQuick',
                                           iterfield=['participant_id', 'session_id', 'moving_image'],
                                        interface=nutil.Function(input_names=['fix_image', 'moving_image', 'participant_id', 'session_id'],
                                                              output_names=['image_warped', 'affine_matrix', 'warp',
                                                                            'inverse_warped', 'inverse_warp'],
                                                              function=ants_registration_syn_quick))
    antsRegistrationSyNQuick.inputs.fix_image = ref_template

    rename_file = npe.MapNode(name='rename_file',
                            iterfield=['subject', 'session'],
                               interface=nutil.Function(
                                   function=rename_file,
                                   input_names=['subject', 'session'],
                                   output_names=['output_file_name']))

    cropnifti = npe.MapNode(name='cropnifti',
                            iterfield=['input_img'],
                               interface=nutil.Function(
                                   function=crop_nifti,
                                   input_names=['input_img', 'ref_img'],
                                   output_names=['output_img', 'crop_template']))
    cropnifti.inputs.ref_img = ref_template

    ## histogram-based intensity normalization
    intensitynorm = npe.MapNode(name='intensitynormalization',
                                           iterfield=['input_img', 'crop_template'],
                                        interface=nutil.Function(input_names=['image_dimension', 'crop_template', 'input_img'],
                                                              output_names=['output_img'],
                                                              function=ants_histogram_intensity_normalization))
    intensitynorm.inputs.image_dimension = 3


    outputnode = npe.Node(nutil.IdentityInterface(
        fields=['out_file_inn', 'out_file_crop', 'out_file_reg']),
        name='outputnode')

    # get the information for datasinker.
    get_identifiers = npe.MapNode(nutil.Function(
        input_names=['participant_id', 'session_id', 'caps_directory'], output_names=['base_directory', 'subst_tuple_list', 'regexp_substitutions'],
        function=get_subid_sesid_datasink), iterfield=['participant_id', 'session_id'], name='get_subid_sesid_datasink')
    get_identifiers.inputs.caps_directory = caps_directory

    ### datasink
    datasink = npe.MapNode(nio.DataSink(infields=['out_file_inn', 'out_file_crop', 'out_file_reg']), name='datasinker',
                          iterfield=['out_file_inn', 'out_file_crop', 'out_file_reg', 'base_directory', 'substitutions', 'regexp_substitutions'])
    datasink.inputs.remove_dest_dir = True


    wf = npe.Workflow(name='t1w_preprocessing_dl')
    wf.base_dir = working_directory

    wf.connect([
                (inputnode, get_subject_session_list, [('bids_directory', 'input_dir')]),
                (inputnode, get_subject_session_list, [('tsv', 'ss_file')]),

                (inputnode, datagrabbernode, [('bids_directory', 'input_dir')]),
                (get_subject_session_list, datagrabbernode, [('subjects', 'subject_list')]),
                (get_subject_session_list, datagrabbernode, [('sessions', 'session_list')]),

                (get_subject_session_list, rename_file, [('subjects', 'subject')]),
                (get_subject_session_list, rename_file, [('sessions', 'session')]),

                (datagrabbernode, n4biascorrection, [('anat_t1', 'input_image')]),
                (rename_file, n4biascorrection, [('output_file_name', 'output_image')]),

                (n4biascorrection, antsRegistrationSyNQuick, [('output_image', 'moving_image')]),
                (get_subject_session_list, antsRegistrationSyNQuick, [('subjects', 'participant_id')]),
                (get_subject_session_list, antsRegistrationSyNQuick, [('sessions', 'session_id')]),

                (antsRegistrationSyNQuick, cropnifti, [('image_warped', 'input_img')]),

                (cropnifti, intensitynorm, [('output_img', 'input_img')]),
                (cropnifti, intensitynorm, [('crop_template', 'crop_template')]),

                ## datasink
                # Saving files with datasink:
                (get_subject_session_list, get_identifiers, [('subjects', 'participant_id')]),
                (get_subject_session_list, get_identifiers, [('sessions', 'session_id')]),

                (get_identifiers, datasink, [('base_directory', 'base_directory')]),
                (get_identifiers, datasink, [('subst_tuple_list', 'substitutions')]),
                (get_identifiers, datasink, [('regexp_substitutions', 'regexp_substitutions')]),
                # datasink to save outputs
                # smoothed dti maps
                (intensitynorm, datasink, [('output_img', 'out_file_inn')]),
                (intensitynorm, outputnode, [('output_img', 'out_file_inn')]),
                (cropnifti, datasink, [('output_img', 'out_file_crop')]),
                (cropnifti, outputnode, [('output_img', 'out_file_crop')]),
                (antsRegistrationSyNQuick, datasink, [('image_warped', 'out_file_reg')]),
                (antsRegistrationSyNQuick, outputnode, [('image_warped', 'out_file_reg')]),
                ])


    return wf



