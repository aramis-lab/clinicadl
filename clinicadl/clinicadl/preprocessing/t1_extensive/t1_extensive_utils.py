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
