"""
This file contains all methods needed to perform the quality check procedure after t1-linear preprocessing.
"""


def quality_check(caps_dir, tsv_path, output_path, threshold=0.5):
    import sys
    from os import path
    import torch
    import pandas as pd

    if path.splitext(output_path)[1] != ".tsv":
        raise ValueError("Please provide an output path to a tsv file")

    # Load QC model
    script_dir = path.dirname(sys.argv[0])
    model = torch.load(path.join(script_dir, "model", "resnet18.pth"))
    model.eval()

    # Load DataFrame
    df = pd.read_csv(tsv_path, sep='\t')
    if ('session_id' not in list(df.columns.values)) or ('participant_id' not in list(df.columns.values)):
        raise Exception("the data file is not in the correct format."
                        "Columns should include ['participant_id', 'session_id']")

    columns = ['participant_id', 'session_id', 'pass_probability', 'pass']
    qc_df = pd.DataFrame(columns=columns)
    softmax = torch.nn.Softmax(dim=1)

    for _, row in df.iterrows():
        subject = row['participant_id']
        session = row['session_id']
        img_path = path.join(caps_dir, 'subjects', subject, session, 't1_linear',
                             '%s_%s_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'
                             % (subject, session))

        inputs = load_nifti_images(img_path)
        inputs = torch.cat(inputs).unsqueeze_(0)

        outputs = softmax.forward(model(inputs))
        pass_prob = outputs.data[0, 1]

        if pass_prob >= threshold:
            row = [[subject, session, pass_prob.item(), True]]
        else:
            row = [[subject, session, pass_prob.item(), False]]

        row_df = pd.DataFrame(row, columns=columns)
        qc_df = qc_df.append(row_df)

    qc_df.to_csv(output_path, sep='\t', index=False)


def load_nifti_images(image_path):
    import nibabel as nib
    import numpy as np
    import torch
    from skimage import transform

    image = nib.load(image_path)
    sample = np.array(image.get_data())

    # normalize input
    _min = np.min(sample)
    _max = np.max(sample)
    sample = (sample-_min)*(1.0/(_max-_min))-0.5
    sz = sample.shape
    input_images = [
        sample[:, :, int(sz[2]/2)],
        sample[int(sz[0] / 2), :, :],
        sample[:, int(sz[1]/2), :]
    ]

    output_images = [
        np.zeros((224, 224),),
        np.zeros((224, 224)),
        np.zeros((224, 224))
    ]

    # flip, resize and crop
    for i in range(3):
        # try the dimension of input_image[i]
        # rotate the slice with 90 degree, I don't know why, but read from nifti file, the img has been rotated, thus we do not have the same direction with the pretrained model

        if len(input_images[i].shape) == 3:
            slice = np.reshape(
                input_images[i], (input_images[i].shape[0], input_images[i].shape[1]))
        else:
            slice = input_images[i]

        _scale = min(256.0/slice.shape[0], 256.0/slice.shape[1])
        # slice[::-1, :] is to flip the first axis of image
        slice = transform.rescale(
            slice[::-1, :], _scale, mode='constant', clip=False)

        sz = slice.shape
        # pad image
        dummy = np.zeros((256, 256),)
        dummy[int((256-sz[0])/2): int((256-sz[0])/2)+sz[0],
              int((256-sz[1])/2): int((256-sz[1])/2)+sz[1]] = slice

        # rotate and flip the image back to the right direction for each view, if the MRI was read by nibabel
        # it seems that this will rotate the image 90 degree with counter-clockwise direction and then flip it horizontally
        output_images[i] = np.flip(
            np.rot90(dummy[16:240, 16:240]), axis=1).copy()

    return [torch.from_numpy(i).float().unsqueeze_(0) for i in output_images]
