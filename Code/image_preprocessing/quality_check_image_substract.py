# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 14:17:41 2017

@author: Junhao WEN
"""

def quality_check_image_similarity(caps_directory, tsv, ref_template, working_directory=None):
    """
    This is a function to do a raw quality check for the preprocessed image. To mention, the preprocessing pipeline of DL includes:
        1) N4 bias correction (Ants)
        2) linear registration to MNI (MNI icbm152 nlinear sym template) (ANTS)
        3) cropping the background to save the computational power

    To note, we use mutual information to quantify the similarity between the target and template MRI.

    Args:

    Returns: A tsv with an order based on the intensity differences between each individual and the template.

    """


    import nipype.interfaces.utility as nutil
    import nipype.pipeline.engine as npe
    import tempfile
    from nipype.algorithms.metrics import Similarity
    from T1_preprocessing_utils import get_caps_list, rank_mean

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    inputnode = npe.Node(nutil.IdentityInterface(
        fields=['caps_directory', 'tsv', 'ref_template']),
        name='inputnode')
    inputnode.inputs.caps_directory = caps_directory
    inputnode.inputs.tsv = tsv
    inputnode.inputs.ref_template = ref_template

    get_caps_list = npe.Node(name='get_caps_list',
                               interface=nutil.Function(
                                   function=get_caps_list,
                                   input_names=['caps_directory', 'tsv'],
                                   output_names=['caps_intensity_nor_list']))

    img_similarity = npe.MapNode(name='img_similarity',
                                           iterfield=['volume1'],
                                 interface=Similarity())
    img_similarity.inputs.metric='mi'

    ### rand the mean intensity
    rank_mean = npe.Node(name='rank_mean',
                               interface=nutil.Function(
                                   function=rank_mean,
                                   input_names=['similarity', 'tsv', 'caps_directory'],
                                   output_names=['result_tsv']))

    wf = npe.Workflow(name='quality_check_dl')
    wf.base_dir = working_directory

    wf.connect([
                (inputnode, get_caps_list, [('caps_directory', 'caps_directory')]),
                (inputnode, get_caps_list, [('tsv', 'tsv')]),

                (get_caps_list, img_similarity, [('caps_intensity_nor_list', 'volume1')]),
                (inputnode, img_similarity, [('ref_template', 'volume2')]),


                (inputnode, rank_mean, [('tsv', 'tsv')]),
                (inputnode, rank_mean, [('caps_directory', 'caps_directory')]),
                (img_similarity, rank_mean, [('similarity', 'similarity')])
                ])


    return wf