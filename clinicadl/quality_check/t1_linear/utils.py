"""
Copied from https://github.com/vfonov/darq
"""

from pathlib import Path

import nibabel as nib
import torch
from torch.utils.data import Dataset

from clinicadl.prepare_data.prepare_data_utils import compute_folder_and_file_type
from clinicadl.utils.clinica_utils import clinicadl_file_reader, linear_nii


class QCDataset(Dataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(
        self,
        img_dir: Path,
        data_df,
        use_extracted_tensors=False,
        use_uncropped_image=True,
    ):
        """
        Args:
            img_dir (string): Directory of all the images.
            data_df (DataFrame): Subject and session list.

        """
        from clinicadl.utils.caps_dataset.data import MinMaxNormalization

        self.img_dir = img_dir
        self.df = data_df
        self.use_extracted_tensors = use_extracted_tensors
        self.use_uncropped_image = use_uncropped_image

        if ("session_id" not in list(self.df.columns.values)) or (
            "participant_id" not in list(self.df.columns.values)
        ):
            raise Exception(
                "The data file is not in the correct format."
                "Columns should include ['participant_id', 'session_id']"
            )

        self.normalization = MinMaxNormalization()

        self.preprocessing_dict = {
            "preprocessing": "t1-linear",
            "mode": "image",
            "use_uncropped_image": use_uncropped_image,
            "file_type": linear_nii("T1w", use_uncropped_image),
            "use_tensor": use_extracted_tensors,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        subject = self.df.loc[idx, "participant_id"]
        session = self.df.loc[idx, "session_id"]

        if self.use_extracted_tensors:
            file_type = self.preprocessing_dict["file_type"]
            file_type["pattern"] = file_type["pattern"].replace(".nii.gz", ".pt")
            image_output = clinicadl_file_reader(
                [subject], [session], self.img_dir, file_type
            )[0]
            image_path = Path(image_output[0])
            image_filename = image_path.name
            folder, _ = compute_folder_and_file_type(self.preprocessing_dict)
            image_dir = (
                self.img_dir
                / "subjects"
                / subject
                / session
                / "deeplearning_prepare_data"
                / "image_based"
                / folder
            )

            image_path = image_dir / image_filename
            image = torch.load(image_path)
            image = self.pt_transform(image)
        else:
            image_path = clinicadl_file_reader(
                [subject],
                [session],
                self.img_dir,
                linear_nii("T1w", self.use_uncropped_image),
            )[0]
            image = nib.load(image_path[0])
            image = self.nii_transform(image)

        sample = {"image": image, "participant_id": subject, "session_id": session}

        return sample

    @staticmethod
    def nii_transform(image):
        import numpy as np
        import torch
        from skimage import transform

        sample = np.array(image.get_fdata())

        # normalize input
        _min = np.min(sample)
        _max = np.max(sample)
        sample = (sample - _min) * (1.0 / (_max - _min)) - 0.5

        sz = sample.shape
        input_images = [
            sample[:, :, int(sz[2] / 2)],
            sample[int(sz[0] / 2), :, :],
            sample[:, int(sz[1] / 2), :],
        ]

        output_images = [
            np.zeros(
                (224, 224),
            ),
            np.zeros((224, 224)),
            np.zeros((224, 224)),
        ]

        # flip, resize and crop
        for i in range(3):
            # try the dimension of input_image[i]
            # rotate the slice with 90 degree, I don't know why, but read from
            # nifti file, the img has been rotated, thus we do not have the same
            # direction with the pretrained model

            if len(input_images[i].shape) == 3:
                slice = np.reshape(
                    input_images[i],
                    (input_images[i].shape[0], input_images[i].shape[1]),
                )
            else:
                slice = input_images[i]

            _scale = min(256.0 / slice.shape[0], 256.0 / slice.shape[1])
            # slice[::-1, :] is to flip the first axis of image
            slice = transform.rescale(
                slice[::-1, :], _scale, mode="constant", clip=False
            )

            sz = slice.shape
            # pad image
            dummy = np.zeros(
                (256, 256),
            )
            dummy[
                int((256 - sz[0]) / 2) : int((256 - sz[0]) / 2) + sz[0],
                int((256 - sz[1]) / 2) : int((256 - sz[1]) / 2) + sz[1],
            ] = slice

            # rotate and flip the image back to the right direction for each view, if the MRI was read by nibabel
            # it seems that this will rotate the image 90 degree with
            # counter-clockwise direction and then flip it horizontally
            output_images[i] = np.flip(np.rot90(dummy[16:240, 16:240]), axis=1).copy()

        return torch.cat(
            [torch.from_numpy(i).float().unsqueeze_(0) for i in output_images]
        ).unsqueeze_(0)

    def pt_transform(self, image):
        import numpy as np
        from torch.nn.functional import interpolate

        image = self.normalization(image) - 0.5
        image = image[0, :, :, :]
        sz = image.shape
        input_images = [
            image[:, :, int(sz[2] / 2)],
            image[int(sz[0] / 2), :, :],
            image[:, int(sz[1] / 2), :],
        ]

        output_images = [
            np.zeros(
                (224, 224),
            ),
            np.zeros((224, 224)),
            np.zeros((224, 224)),
        ]

        # flip, resize and crop
        for i in range(3):
            # try the dimension of input_image[i]
            # rotate the slice with 90 degree, I don't know why, but read from
            # nifti file, the img has been rotated, thus we do not have the same
            # direction with the pretrained model

            if len(input_images[i].shape) == 3:
                slice = np.reshape(
                    input_images[i],
                    (input_images[i].shape[0], input_images[i].shape[1]),
                )
            else:
                slice = input_images[i]

            _scale = min(256.0 / slice.shape[0], 256.0 / slice.shape[1])
            # slice[::-1, :] is to flip the first axis of image
            slice = interpolate(
                torch.flip(slice, (0,)).unsqueeze(0).unsqueeze(0), scale_factor=_scale
            )
            slice = slice[0, 0, :, :]

            sz = slice.shape
            # pad image
            dummy = np.zeros(
                (256, 256),
            )
            dummy[
                int((256 - sz[0]) / 2) : int((256 - sz[0]) / 2) + sz[0],
                int((256 - sz[1]) / 2) : int((256 - sz[1]) / 2) + sz[1],
            ] = slice

            # rotate and flip the image back to the right direction for each view, if the MRI was read by nibabel
            # it seems that this will rotate the image 90 degree with
            # counter-clockwise direction and then flip it horizontally
            output_images[i] = np.flip(np.rot90(dummy[16:240, 16:240]), axis=1).copy()

        return torch.cat(
            [torch.from_numpy(i).float().unsqueeze_(0) for i in output_images]
        ).unsqueeze_(0)

    @staticmethod
    def get_padding(image):
        max_w = 256
        max_h = 256

        imsize = image.shape
        h_padding = (max_w - imsize[1]) / 2
        v_padding = (max_h - imsize[0]) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5

        padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))

        return padding
