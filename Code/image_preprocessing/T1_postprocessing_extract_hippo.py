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

def postprocessing_t1w_extract_hippo(caps_directory, tsv, ref_template, working_directory=None):
    """
    This is a postprocessing pipeline to prepare the slice-level and patch-level data from the whole MRI and save them
    on disk, so that to facilitate the training process:
        - For slice-level CNN, all slices were extracted from the whole MRI from three different axis. The first and last
          15 slice were discarded due to the lack of information.
        - For patch-level CNN, the 3D patch (with specific patch size) were extracted by a 3D window.

    :param caps_directory: CAPS directory where stores the output of preprocessing
    :param tsv: subject list file containing the participant_id and session_id
    :param patch_size: the size for extracted 3D patches
    :param stride_size: the sliding siez of the 3D windows
    :param working_directory: working directory to store the intermediate files
    :return:
    """

    import nipype.interfaces.io as nio
    import nipype.interfaces.utility as nutil
    import nipype.pipeline.engine as npe
    import tempfile
    from T1_postprocessing_utils import get_caps_t1, crop_niftii_hippo

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    inputnode = npe.Node(nutil.IdentityInterface(
        fields=['caps_directory', 'tsv', 'ref_template']),
        name='inputnode')
    inputnode.inputs.caps_directory = caps_directory
    inputnode.inputs.tsv = tsv
    inputnode.inputs.ref_template = ref_template

    get_subject_session_list = npe.Node(name='get_subject_session_list',
                               interface=nutil.Function(
                                   function=get_caps_t1,
                                   input_names=['caps_directory', 'tsv'],
                                   output_names=['preprocessed_T1']))

    ## extract the patches.
    cropnifti_hippo = npe.MapNode(name='cropnifti',
                            iterfield=['input_img'],
                               interface=nutil.Function(
                                   function=crop_niftii_hippo,
                                   input_names=['input_img', 'ref_img'],
                                   output_names=['output_img', 'crop_template']))
    cropnifti.inputs.ref_img = ref_template

    outputnode = npe.Node(nutil.IdentityInterface(
        fields=['preprocessed_T1']),
        name='outputnode')

    wf = npe.Workflow(name='t1w_postprocessing_dl_extract_hippo')
    wf.base_dir = working_directory

    wf.connect([
            (inputnode, get_subject_session_list, [('tsv', 'tsv')]),
            (inputnode, get_subject_session_list, [('caps_directory', 'caps_directory')]),

            (get_subject_session_list, cropnifti_hippo, [('preprocessed_T1', 'input_img')]),
            (cropnifti_hippo, outputnode, [('preprocessed_T1', 'preprocessed_T1')]),
    ])
