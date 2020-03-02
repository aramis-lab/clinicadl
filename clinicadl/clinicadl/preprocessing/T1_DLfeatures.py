# -*- coding: utf-8 -*-


def extract_DL_features_t1w(caps_directory, 
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

    import nipype.interfaces.io as nio
    import nipype.interfaces.utility as nutil
    import nipype.pipeline.engine as npe
    import tempfile
    from clinica.utils.inputs import check_caps_folder
    from clinica.utils.filemanip import get_subject_id
    from clinica.utils.participant import get_subject_session_list
    from clinica.utils.exceptions import ClinicaBIDSError, ClinicaException
    from clinica.utils.inputs import clinica_file_reader
    from .T1_postprocessing_utils import (extract_slices,
            extract_patches, save_as_pt)
    from .T1_extractdl_utils import (container_from_filename, get_data_datasink)
    from nipype.interfaces.io import DataSink
    from clinica.utils.nipype import fix_join
    
    T1W_LINEAR =  {
            'pattern': '*space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz',
            'description': 'T1W Image registered using T1_Linear'
            }
    
    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    check_caps_folder(caps_directory)
    is_bids_dir = False
    
    sessions, subjects = get_subject_session_list(
            caps_directory,
            tsv,
            is_bids_dir,
            False,
            working_directory
            )

    # Use hash instead of parameters for iterables folder names
    # Otherwise path will be too long and generate OSError
    from nipype import config
    cfg = dict(execution={'parameterize_dirs': False})
    config.update_config(cfg)


    # Inputs from anat/ folder
    # ========================
    # T1w file:
    try:
        t1w_files = clinica_file_reader(subjects,
                sessions,
                caps_directory,
                T1W_LINEAR)
    except ClinicaException as e:
        err = 'Clinica faced error(s) while trying to read files in your CAPS directory.\n' + str(e)
        raise ClinicaBIDSError(err)

    def get_input_fields():
        """"Specify the list of possible inputs of this pipelines.
        Returns:
        A list of (string) input fields name.
        """
        return ['t1w']

    print(t1w_files)

    read_node = npe.Node(name="ReadingFiles",
           iterables=[
               ('t1w', t1w_files),
               ],
           synchronize=True,
           interface=nutil.IdentityInterface(
               fields=get_input_fields())
           )

    image_id_node = npe.Node(
           interface=nutil.Function(
               input_names=['bids_or_caps_file'],
               output_names=['image_id'],
               function= get_subject_id),
           name='ImageID'
           )

    ## The core (processing) nodes
    
    

    ## save nii.gz into classifiers .pt format.
    save_as_pt = npe.MapNode(
           name='save_as_pt',
           iterfield=['input_img'],
           interface=nutil.Function(
               function=save_as_pt,
               input_names=['input_img'],
               output_names=['output_file'] 
               )
           )

    ## extract the slices from 3 directions.
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

    ## extract the patches.
    extract_patches = npe.MapNode(
            name='extract_patches',
            iterfield=['preprocessed_T1'],
            interface=nutil.Function(
                function=extract_patches,
                input_names=['preprocessed_T1', 'patch_size', 'stride_size'],
                output_names=['preprocessed_T1']
                )
            )

    extract_patches.inputs.patch_size = patch_size
    extract_patches.inputs.stride_size = stride_size

    outputnode = npe.Node(
            nutil.IdentityInterface(
                fields=['preprocessed_T1']),
            name='outputnode'
            )

    get_ids = npe.Node(
            interface=nutil.Function(
                input_names=['image_id'],
                output_names=['image_id_out', 'subst_ls'],
                function=get_data_datasink),
            name="GetIDs")
    # Find container path from t1w filename
    # =====================================
    container_path = npe.Node(
            nutil.Function(
                input_names=['bids_or_caps_filename'],
                output_names=['container'],
                function=container_from_filename),
            name='ContainerPath')

    write_node = npe.Node(
            name="WriteCaps",
            interface=DataSink()
            )
    write_node.inputs.base_directory = caps_directory
    write_node.inputs.parameterization = False

    wf = npe.Workflow(name='t1_dl_features', base_dir=working_directory)

    wf.connect([
       (read_node, image_id_node, [('t1w', 'bids_or_caps_file')]),
       (read_node, container_path, [('t1w', 'bids_or_caps_filename')]),
       (read_node, save_as_pt, [('t1w', 'input_img')])
       ])
    
    if extract_method == 'slice':
        wf.connect([
            (save_as_pt, extract_slices, [('output_file', 'preprocessed_T1')]),
            (extract_slices, write_node, [('preprocessed_T1', '@preprocessed_T1')])
            ])
    elif extract_method == 'patch':
        wf.connect([
            (save_as_pt, extract_patches, [('output_file', 'preprocessed_T1')]),
            (extract_patches, write_node, [('preprocessed_T1', '@preprocessed_T1')])
            ])
    else:
        wf.connect([
            (save_as_pt, extract_slices, [('output_file', 'preprocessed_T1')]),
            (extract_slices, write_node, [('preprocessed_T1', '@preprocessed_T1')])

            (save_as_pt, extract_patches, [('output_file', 'preprocessed_T1')]),
            (extract_patches, write_node, [('preprocessed_T1', '@preprocessed_T1')])
            ])
        
    wf.connect([    
        # Connect to DataSink
        (container_path, write_node, [(('container', fix_join, 't1_linear'), 'container')]),
        (image_id_node, get_ids, [('image_id', 'image_id')]),
        (get_ids, write_node, [('image_id_out', '@image_id')]),
        (get_ids, write_node, [('subst_ls', 'substitutions')]),
        (save_as_pt, write_node, [('output_file', '@output_pt_file')])    
    ])

    return wf
