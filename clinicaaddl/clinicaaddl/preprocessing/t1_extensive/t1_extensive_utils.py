# coding: utf8


def get_caps_filename(norm_t1w):
    """Generate output CAPS filename from input CAPS filename

    Args:
        norm_t1w: T1w in Ixi549Space space
            (output from t1-volume-tissue-segmentation)

    Returns:
        Filename (skull-stripped T1w in Ixi549Space space) for t1-extensive pipeline
    """
    from nipype.utils.filemanip import split_filename

    orig_dir, base, ext = split_filename(norm_t1w)
    # <base>: <participant_id>_<session_id>*_space-Ixi549Space_T1w
    skull_stripped_t1w = (
        base.replace(
            "_space-Ixi549Space_T1w", "_space-Ixi549Space_desc-SkullStripped_T1w"
        )
        + ext
    )

    return skull_stripped_t1w


def apply_binary_mask(input_img, binary_img, output_filename):
    """Apply binary mask to input_img.

    Args:
        input_img: Image with same header than binary_image
        binary_img: Binary image
        output_filename: Output filename

    Returns:
        input_img*binary_img
    """
    import os
    import nibabel as nib

    original_image = nib.load(input_img)
    mask = nib.load(binary_img)

    data = original_image.get_data() * mask.get_data()

    masked_image_path = os.path.join(os.getcwd(), output_filename)
    masked_image = nib.Nifti1Image(
        data, original_image.affine, header=original_image.header
    )
    nib.save(masked_image, masked_image_path)
    return masked_image_path


def get_file_from_server(remote_file, cache_path=None):
    """
    Download file from server

    Args:
        remote_file (str): RemoteFileStructure defined in clinica.utils.inputs
        cache_path (str): (default: ~/.cache/clinica/data)

    Returns:
        Path to downloaded file.

    Note:
        This function will be in Clinica.
    """
    import os
    from pathlib import Path
    from clinica.utils.stream import cprint
    from clinica.utils.inputs import fetch_file

    home = str(Path.home())
    if cache_path:
        cache_clinica = os.path.join(home, ".cache", cache_path)
    else:
        cache_clinica = os.path.join(home, ".cache", "clinica", "data")
    if not (os.path.exists(cache_clinica)):
        os.makedirs(cache_clinica)

    local_file = os.path.join(cache_clinica, remote_file.filename)

    if not (os.path.exists(local_file)):
        try:
            local_file = fetch_file(remote_file, cache_clinica)
        except IOError as err:
            cprint(
                f"Unable to download {remote_file.filename} from {remote_file.url}: {err}"
            )

    return local_file
