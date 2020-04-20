# -*- coding: utf-8 -*-


__author__ = "Junhao Wen, Mauricio Diaz"
__copyright__ = "Copyright 2018-2020 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.0.1"
__maintainer__ = "Mauricio Diaz"
__email__ = "mauricio.diaz@inria.fr"
__status__ = "Development"


def postprocessing_t1w(
        caps_directory,
        tsv,
        patch_size,
        stride_size,
        working_directory=None,
        extract_method='slice',
        slice_direction=0,
        slice_mode='original'):
    """
    This is a postprocessing pipeline to prepare the slice-level and
    patch-level data from the entire  MRI and save them on disk. This
    facilitate the training process:
        - For slice-level CNN, all slices were extracted from the entire
          MRI from three different axis. The first and last 15 slice were
          discarded due to the lack of information.
        - For patch-level CNN, the 3D patch (with specific patch size)
          were extracted by a 3D window.

    Parameters
    ----------

    caps_directory: str
      CAPS directory where stores the output of preprocessing.
    tsv: str
      TVS file with the subject list (participant_id and session_id).
    patch_size: int
      Size for extracted 3D patches.
    stride_size: int
      Sliding size window of the slice feature.
    working_directory: str
      Folder containing a temporary space to save intermediate results.


    Returns
    -------
    wf: class nipype.pipeline.engine.workflows.Workflow
      A class du type nypipe workflow to control, setup, and execute a process
      as a nypipe pipeline.

    """

    # test.py
    print(__name__)

#    try:
#        # Trying to find module in the parent package
#        import preprocessing.T1_postprocessing_utils
#        print(T1_postprocessing_utils.debug)
#        del T1_postprocessing_utils
#    except ModuleNotFoundError:
#            print('Relative import failed')
#
#    try:
#        # Trying to find module on sys.path
#        import T1_postprocessing_utils
#        print(T1_postprocessing_utils.debug)
#    except ModuleNotFoundError:
#        print('Absolute import failed')
#

    import nipype.interfaces.io as nio
    import nipype.interfaces.utility as nutil
    import nipype.pipeline.engine as npe
    import tempfile
    from .T1_postprocessing_utils import (get_caps_t1,
                                          extract_slices,
                                          extract_patches,
                                          save_as_pt)

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    inputnode = npe.Node(
            nutil.IdentityInterface(
                fields=['caps_directory', 'tsv', 'patch_size', 'stride_size']),
            name='inputnode'
            )

    inputnode.inputs.caps_directory = caps_directory
    inputnode.inputs.tsv = tsv
    inputnode.inputs.patch_size = patch_size
    inputnode.inputs.stride_size = stride_size

    get_subject_session_list = npe.Node(
            name='get_subject_session_list',
            interface=nutil.Function(
                function=get_caps_t1,
                input_names=['caps_directory', 'tsv'],
                output_names=['preprocessed_T1']
                )
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

    # extract the slices from 3 directions.
    extract_slices = npe.MapNode(
            name='extract_slices',
            iterfield=['preprocessed_T1'],
            interface=nutil.Function(
                function=extract_slices,
                input_names=[
                    'preprocessed_T1', 'slice_direction',
                    'slice_mode'
                    ],
                output_names=['preprocessed_T1']
                )
            )

    extract_slices.inputs.slice_direction = slice_direction
    extract_slices.inputs.slice_mode = slice_mode

    # extract the patches.
    extract_patches = npe.MapNode(
            name='extract_patches',
            iterfield=['preprocessed_T1'],
            interface=nutil.Function(
                function=extract_patches,
                input_names=['preprocessed_T1', 'patch_size', 'stride_size'],
                output_names=['preprocessed_T1']
                )
            )

    outputnode = npe.Node(
            nutil.IdentityInterface(
                fields=['preprocessed_T1']),
            name='outputnode'
            )

    wf = npe.Workflow(name='t1w_postprocessing_dl')
    wf.base_dir = working_directory

    if extract_method == 'slice':
        wf.connect(
                [
                    (inputnode, get_subject_session_list, [('tsv', 'tsv')]),
                    (inputnode, get_subject_session_list, [
                        ('caps_directory', 'caps_directory')
                        ]),

                    (get_subject_session_list, save_as_pt, [
                        ('preprocessed_T1', 'input_img')
                        ]),
                    (save_as_pt, extract_slices, [
                        ('output_file', 'preprocessed_T1')
                        ]),
                    (extract_slices, outputnode, [
                        ('preprocessed_T1', 'preprocessed_T1')
                        ]),
                    ]
                )
    elif extract_method == 'patch':
        wf.connect([
            (inputnode, get_subject_session_list, [('tsv', 'tsv')]),
            (inputnode, get_subject_session_list, [('caps_directory', 'caps_directory')]),

            (get_subject_session_list, save_as_pt, [('preprocessed_T1', 'input_img')]),
            (save_as_pt, extract_patches, [('output_file', 'preprocessed_T1')]),
            (inputnode, extract_patches, [('patch_size', 'patch_size')]),
            (inputnode, extract_patches, [('stride_size', 'stride_size')]),
        ])
    else:
        wf.connect([
                    (inputnode, get_subject_session_list, [('tsv', 'tsv')]),
                    (inputnode, get_subject_session_list, [('caps_directory', 'caps_directory')]),

                    (get_subject_session_list, save_as_pt, [('preprocessed_T1', 'input_img')]),
                    (save_as_pt, extract_slices, [('output_file', 'preprocessed_T1')]),

                    (get_subject_session_list, save_as_pt, [('preprocessed_T1', 'input_img')]),
                    (save_as_pt, extract_patches, [('output_file', 'preprocessed_T1')]),
                    (inputnode, extract_patches, [('patch_size', 'patch_size')]),
                    (inputnode, extract_patches, [('stride_size', 'stride_size')]),

                    (extract_slices, outputnode, [('preprocessed_T1', 'preprocessed_T1')]),
                    ])

    return wf
