# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 14:17:41 2017

@author: Junhao WEN
"""

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


def postprocessing_t1w_extract_hippo(caps_directory, tsv, working_directory=None, hemi='right'):
    """
    This is a postprocessing pipeline to prepare the slice-level and patch-level data from the whole MRI and save them
    on disk, so that to facilitate the training process:
        - For slice-level CNN, all slices were extracted from the whole MRI from three different axis. The first and last
          15 slice were discarded due to the lack of information.
        - For patch-level CNN, the 3D patch (with specific patch size) were extracted by a 3D window.

    :param caps_directory: CAPS directory where stores the output of preprocessing
    :param tsv: subject list file containing the participant_id and session_id
    :param hemi: chooses which hippocampus is extracted (left or right)
    :param working_directory: working directory to store the intermediate files
    :return:
    """

    from nipype.interfaces.freesurfer import MRIConvert
    import nipype.interfaces.utility as nutil
    import nipype.pipeline.engine as npe
    import nipype.interfaces.io as nio
    import tempfile
    from .T1_postprocessing_extract_hippo_utils import get_caps_t1, save_as_pt, compress_nii, get_subid_sesid_datasink

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    inputnode = npe.Node(nutil.IdentityInterface(
        fields=['caps_directory', 'tsv']),
        name='inputnode')
    inputnode.inputs.caps_directory = caps_directory
    inputnode.inputs.tsv = tsv

    get_subject_session_list = npe.Node(
            name='get_subject_session_list',
            interface=nutil.Function(
                function=get_caps_t1,
                input_names=['caps_directory', 'tsv'],
                output_names=[
                    'preprocessed_T1',
                    'cropped_hipp_file_name',
                    'participant_id',
                    'session_id',
                    'preprocessed_T1_folder'
                    ]
                )
            )

    # extract the hippocampus.
    hippocampus_patches = npe.MapNode(
            name='hippocampus_patches',
            iterfield=['in_file', 'out_file'],
            interface=MRIConvert()
            )

    hippocampus_patches.inputs.out_type = 'nii'

    # TODO, to decide the position of hippocampus of each hemisphere
    if hemi == 'left':
        hippocampus_patches.inputs.crop_center = (61, 96, 68)  # the center of the right and left hippocampus
        hippocampus_patches.inputs.crop_size = (50, 50, 50)  # the output cropped hippocampus size
    else:
        hippocampus_patches.inputs.crop_center = (109, 96, 68)  # the center of the right and right hippocampus
        hippocampus_patches.inputs.crop_size = (50, 50, 50)  # the output cropped hippocampus size

    # zip the result imgs
    # in the newest version of nipype for MRIConvert, it seems that they can be saved directly as nii.gz
    zip_hippocampus = npe.MapNode(
            name='zip_hippocampus',
            interface=nutil.Function(
                input_names=['in_file'],
                output_names=['out_file'],
                function=compress_nii
                ),
            iterfield=['in_file']
            )

    # save nii.gz into classifiers .pt format.
    save_as_pt = npe.MapNode(
            name='save_as_pt',
            iterfield=['input_img'],
            interface=nutil.Function(
                function=save_as_pt,
                input_names=['input_img'],
                output_names=['output_file']
                )
            )

    # get the information for datasinker.
    get_identifiers = npe.MapNode(nutil.Function(
        input_names=['participant_id', 'session_id', 'caps_directory', 'hemi'], output_names=['base_directory', 'subst_tuple_list', 'regexp_substitutions'],
        function=get_subid_sesid_datasink), iterfield=['participant_id', 'session_id'], name='get_subid_sesid_datasink')
    get_identifiers.inputs.caps_directory = caps_directory
    get_identifiers.inputs.hemi = hemi

    # datasink
    datasink = npe.MapNode(
            nio.DataSink(
                infields=['output_hippocampus_nii', 'output_hippocampus_pt']
                ),
            name='datasinker',
            iterfield=[
                'output_hippocampus_nii',
                'output_hippocampus_pt',
                'base_directory',
                'substitutions',
                'regexp_substitutions'
                ]
            )

    outputnode = npe.Node(
            nutil.IdentityInterface(
                fields=['output_hippocampus_nii', 'output_hippocampus_pt']
                ),
            name='outputnode')

    wf = npe.Workflow(name='t1w_postprocessing_dl_extract_hippo')
    wf.base_dir = working_directory

    wf.connect([
            (inputnode, get_subject_session_list, [('tsv', 'tsv')]),
            (inputnode, get_subject_session_list, [('caps_directory', 'caps_directory')]),

            (get_subject_session_list, hippocampus_patches, [('preprocessed_T1', 'in_file')]),
            (get_subject_session_list, hippocampus_patches, [('cropped_hipp_file_name', 'out_file')]),
            (hippocampus_patches, zip_hippocampus, [('out_file', 'in_file')]),
            (zip_hippocampus, save_as_pt, [('out_file', 'input_img')]),

            # Saving files with datasink:
            (get_subject_session_list, get_identifiers, [('participant_id', 'participant_id')]),
            (get_subject_session_list, get_identifiers, [('session_id', 'session_id')]),

            (get_identifiers, datasink, [('base_directory', 'base_directory')]),
            (get_identifiers, datasink, [('subst_tuple_list', 'substitutions')]),
            (get_identifiers, datasink, [('regexp_substitutions', 'regexp_substitutions')]),
            (save_as_pt, datasink, [('output_file', 'output_hippocampus_pt')]),
            (zip_hippocampus, datasink, [('out_file', 'output_hippocampus_nii')]),
    ])

    return wf
