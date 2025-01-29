from typing import Tuple, Union

import torchio as tio

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.structures import DataPoint
from clinicadl.utils.typing import PathType

from .base import CapsProcessor, InfoJSON


class ResamplingInfo(InfoJSON):
    """
    To store relevant information on the resampling
    operation.
    """

    spacing: tuple[float, float, float]


class CapsResampler(CapsProcessor):
    """
    Resamples all the images in a CapsDataset to a common physical space.

    Parameters
    ----------
    caps_dataset : CapsDataset
        The CapsDataset on which conversion will be performed.
    """

    def __init__(self, caps_dataset: CapsDataset):
        super().__init__(caps_dataset)
        self.resampler = None
        self.spacing = None
        self._to_canonical = tio.ToCanonical()

    def resample(
        self,
        voxel_spacing: Union[float, Tuple[float, float, float]] = None,
        json_name: PathType = "tensor_conversion",
        n_proc: int = 1,
    ) -> None:
        """_summary_

        Parameters
        ----------
        spacing : Union[float, Tuple[float, float, float]] (optional, default=None)
            _description_
        json_name : PathType (optional, default="tensor_conversion")
            _description_
        n_proc : int (optional, default=1)
            _description_

        Examples
        --------
        >>> resampler = CapsResampler('caps_dir')
        >>> resampler.resample(spacing=1.0)
        # image 'caps_dir/subjects/sub-01/ses-M000/sub-01_ses-M000_T1w.nii.gz'
        # will be resampled in 'caps_dir/subjects/sub-01/ses-M000/sub-01_ses-M000_res-1x1x1_T1w.nii.gz'
        """
        self.resampler = tio.Resample(voxel_spacing)
        self.spacing = (
            voxel_spacing
            if isinstance(voxel_spacing, tuple)
            else (voxel_spacing, voxel_spacing, voxel_spacing)
        )
        self._process_caps(n_proc, json_name)

    @property
    def _store_info(self) -> type[ResamplingInfo]:
        """
        Defines the data structure where to save the information
        on the resampling.
        """
        return ResamplingInfo

    @property
    def _past_participle(self) -> str:
        """
        Past participle corresponding to the processing operation.
        Useful to write warnings or logs.
        """
        return "resampled"

    def _reset(self) -> None:
        """
        Resets the resampler. Nothing to do here.
        """
        pass

    def _gather_info(self) -> ResamplingInfo:
        """
        Gathers all relevant information on the resampling.
        """
        return ResamplingInfo(
            preprocessing=self.preprocessing,
            participants_sessions=self.caps_dataset.get_participant_session_couples(),
            spacing=self.spacing,
        )

    def _process(
        self, data: Union[tio.Image, DataPoint]
    ) -> Union[tio.Image, DataPoint]:
        """
        Converts images to the canonical space (RAS+) and resamples them.
        Accepts a single image or a collection of images related
        to the same (participant, session).
        """
        return self.resampler(self._to_canonical(data))

    def _save_image(self, image: tio.Image) -> None:
        """
        Saves a resampled image.
        The entity 'res' (e.g. 'res-0.9x0.9x0.9') will be added to the name of
        the original image to create the path where the resampled image will
        be stored.
        """
        path = image.path

        body, suffix = path.stem.rsplit("_", maxsplit=1)
        x, y, z = self.spacing
        body += f"_res-{x}x{y}x{z}"
        stem = "_".join([body, suffix])
        new_path = path.with_stem(stem)

        image.save(new_path)

    def _update_caps_dataset(self, info: ResamplingInfo) -> None:
        """
        Updates the state of the Caps Dataset.
        """
        # self.caps_dataset.preprocessing =
