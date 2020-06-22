# -*- coding: utf-8 -*-


def extract_dl_t1w(caps_directory,
                   tsv,
                   working_directory=None,
                   extract_method='image',
                   patch_size=50,
                   stride_size=50,
                   slice_direction=0,
                   slice_mode='single'):
    """ This is a preprocessing pipeline to convert the MRIs in nii.gz format
    into tensor versions (using pytorch format). It also prepares the
    slice-level and patch-level data from the entire MRI and save them on disk.
    This enables the training process:
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
    extract_method:
      Select which extract method will be applied for the outputs:
      - 'image' to convert to PyTorch tensor the complete 3D image,
      - 'patch' to extract 3D volumetric patches and
      - 'slice' to extract 2D slices from the image
    patch_size: int
      Size for extracted 3D patches (only 'patch' method).
    stride_size: int
      Sliding size window of when extracting the patches (only 'patch' method).
    slice_direction: int
      Which direction the slices will be extracted (only 'slice' method):
      - 0: Sagittal plane
      - 1: Coronal plane
      - 2: Axial plane
    slice_mode: str
      Mode how slices are stored (only 'slice' method):
      - single: saves the slice in a single channel,
      - rgb: saves the slice in three identical  channels (red, green, blue)
    working_directory: str
      Folder containing a temporary space to save intermediate results.
    e

    Returns
    -------
    wf: class nipype.pipeline.engine.workflows.Workflow
      A class du type nypipe workflow to control, setup, and execute a process
      as a nypipe pipeline.

    """

    import nipype.interfaces.io as nio
    import nipype.interfaces.utility as nutil
    import nipype.pipeline.engine as npe
    from nipype.interfaces.io import DataSink
    from nipype import config
    import tempfile
    from clinica.utils.inputs import check_caps_folder
    from clinica.utils.filemanip import get_subject_id
    from clinica.utils.participant import get_subject_session_list
    from clinica.utils.exceptions import ClinicaBIDSError, ClinicaException
    from clinica.utils.inputs import clinica_file_reader
    from clinica.utils.nipype import fix_join
    from .T1_preparedl_utils import (extract_slices,
                                     extract_patches,
                                     save_as_pt,
                                     container_from_filename,
                                     get_data_datasink)

    T1W_LINEAR = {'pattern': '*space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz',
                  'description': 'T1W Image registered using T1_Linear'}
    T1W_LINEAR_CROPPED = {'pattern': '*space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz',
                          'description': 'T1W Image registered using T1_Linear and cropped'
                                         '(matrix size 169×208×179, 1 mm isotropic voxels)'}

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    check_caps_folder(caps_directory)
    is_bids_dir = False
    use_session_tsv = False

    sessions, subjects = get_subject_session_list(
            caps_directory,
            tsv,
            is_bids_dir,
            use_session_tsv,
            working_directory
            )

    # Use hash instead of parameters for iterables folder names
    # Otherwise path will be too long and generate OSError
    cfg = dict(execution={'parameterize_dirs': False})
    config.update_config(cfg)

    # Inputs from t1_linear folder
    # ========================
    # T1w file:
    try:
        t1w_files = clinica_file_reader(
                subjects,
                sessions,
                caps_directory,
                T1W_LINEAR_CROPPED
                )
    except ClinicaException as e:
        err = 'Clinica faced error(s) while trying to read files in your CAPS directory.\n' + str(e)
        raise ClinicaBIDSError(err)

    def get_input_fields():
        """"Specify the list of possible inputs of this pipelines.
        Returns:
        A list of (string) input fields name.
        """
        return ['t1w']

    # Read node
    # ----------------------
    read_node = npe.Node(
            name="ReadingFiles",
            iterables=[
                ('t1w', t1w_files),
                ],
            synchronize=True,
            interface=nutil.IdentityInterface(
                fields=get_input_fields())
            )

    # Get subject ID node
    # ----------------------
    image_id_node = npe.Node(
           interface=nutil.Function(
               input_names=['bids_or_caps_file'],
               output_names=['image_id'],
               function=get_subject_id),
           name='ImageID'
           )

    # The processing nodes

    # Node to save MRI in nii.gz format into pytorch .pt format
    # ----------------------
    save_as_pt = npe.MapNode(
           name='save_as_pt',
           iterfield=['input_img'],
           interface=nutil.Function(
               function=save_as_pt,
               input_names=['input_img'],
               output_names=['output_file']
               )
           )

    # Extract slices node (options: 3 directions, mode)
    # ----------------------
    extract_slices = npe.MapNode(
            name='extract_slices',
            iterfield=['input_tensor'],
            interface=nutil.Function(
                function=extract_slices,
                input_names=[
                    'input_tensor', 'slice_direction',
                    'slice_mode'
                    ],
                output_names=['output_file_rgb', 'output_file_original']
                )
            )

    extract_slices.inputs.slice_direction = slice_direction
    extract_slices.inputs.slice_mode = slice_mode

    # Extract patches node (options: patch size and stride size)
    # ----------------------
    extract_patches = npe.MapNode(
            name='extract_patches',
            iterfield=['input_tensor'],
            interface=nutil.Function(
                function=extract_patches,
                input_names=['input_tensor', 'patch_size', 'stride_size'],
                output_names=['output_patch']
                )
            )

    extract_patches.inputs.patch_size = patch_size
    extract_patches.inputs.stride_size = stride_size

    # Output node
    # ----------------------
    outputnode = npe.Node(
            nutil.IdentityInterface(
                fields=['preprocessed_T1']),
            name='outputnode'
            )

    # Find container path from t1w filename
    # ----------------------
    container_path = npe.Node(
            nutil.Function(
                input_names=['bids_or_caps_filename'],
                output_names=['container'],
                function=container_from_filename),
            name='ContainerPath')

    # Write node
    # ----------------------
    write_node = npe.Node(
            name="WriteCaps",
            interface=DataSink()
            )
    write_node.inputs.base_directory = caps_directory
    write_node.inputs.parameterization = False

    subfolder = 'image_based'
    wf = npe.Workflow(name='dl_prepare_data', base_dir=working_directory)

    # Connections
    # ----------------------
    wf.connect([
        (read_node, image_id_node, [('t1w', 'bids_or_caps_file')]),
        (read_node, container_path, [('t1w', 'bids_or_caps_filename')]),
        (read_node, save_as_pt, [('t1w', 'input_img')]),
        (image_id_node, write_node, [('image_id', '@image_id')]),
        # Connect to DataSink
        ])

    if extract_method == 'slice':
        subfolder = 'slice_based'
        wf.connect([
            (save_as_pt, extract_slices, [('output_file', 'input_tensor')]),
            (extract_slices, write_node, [('output_file_rgb', '@slices_rgb_T1')]),
            (extract_slices, write_node, [('output_file_original', '@slices_original_T1')])
            ])
    elif extract_method == 'patch':
        subfolder = 'patch_based'
        wf.connect([
            (save_as_pt, extract_patches, [('output_file', 'input_tensor')]),
            (extract_patches, write_node, [('output_patch', '@patches_T1')])
            ])
    else:
        wf.connect([
            (save_as_pt, write_node, [('output_file', '@output_pt_file')])
            ])

    wf.connect([
        (container_path, write_node, [(
            (
                'container', fix_join,
                'deeplearning_prepare_data', subfolder, 't1_linear'
                ),
            'container')])
        ])

    return wf
