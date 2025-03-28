import copy
from collections import UserString
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torchio as tio

from clinicadl.dictionary.suffixes import PT
from clinicadl.dictionary.words import AFFINE, MASK
from clinicadl.utils.typing import PathType

LabelType = Optional[Union[int, float, tio.LabelMap]]


class Column(UserString):
    """
    Dummy class to store label when it represents a column of a dataframe.
    """

    def __init__(self, name: str):
        self._name = name
        super().__init__(name)

    def __str__(self):
        return f"Column('{self._name}')"


class DataPoint(tio.Subject):
    """
    Dataclass that gathers an image, the associated label, and any other relevant information
    associated to the image.

    It inherits from :py:class:`torchio.Subject`.

    A DataPoint has the following attributes:
        - ``image``: the image, as a :py:class:`torchio.ScalarImage`;
        - ``label``: the label. Either a scalar or a mask, as a :py:class:`torchio.LabelMap`;
        - ``participant``: the id of the subject, as a ``str``;
        - ``session``: the id of the session, as a ``str``.

    You can easily access these elements using the attribute notation:

    .. code-block:: python

        >>> import torchio as tio
        >>> from clinicadl.data.structures import DataPoint
        >>> data = tio.datasets.Colin27()
        >>> datapoint = DataPoint(
            image=data.t1, label=data.brain, participant="sub-colin", session="ses-M000"
        )
        >>> datapoint.session
        'ses-M000'

    Besides, a DataPoint is dictionary-like object. So, you can easily add a key-value pair
    to it:

    .. code-block:: python

        >>> datapoint["age"] = 55
        >>> datapoint["age"]    # the attribute notation won't work here
        55

    However, to add an image or a mask to the DataPoint, prefer :py:func:`~add_image`
    and :py:func:`~add_mask`.

    If all the images and masks of your DataPoint have the same shape, voxel spacing and affine matrix, you can easily
    access them via the attributes :py:attr:`~shape` (or :py:attr:`~spatial_shape` to remove the channel dimension),
    :py:attr:`~spacing` and :py:attr:`~affine` respectively.

    Finally,  you may also be interested in :py:func:`~plot` to plot images inside your DataPoint, and :py:func:`~get_applied_transforms`
    to see the transforms applied to your data.

    As DataPoint is a subclass of :py:class:`torchio.Subject`, you can also used all the other methods it inherits from.

    .. note::
        Any transform used in ClinicaDL must work with DataPoint.

    Parameters
    ----------
    image : Union[torchio.ScalarImage, PathType]
        The image, as a :py:class:`torchio.ScalarImage` or a ``path`` to a NIfTI file.
    label : Optional[Union[float, int, torchio.LabelMap, PathType]]
        The label associated to the image. Can be a ``float`` (regression),
        an ``int`` (classification), a mask (passed as a :py:class:`torchio.LabelMap`
        or a ``path`` to a NIfTI file; for segmentation) or ``None`` if no label (reconstruction).
    participant : str
        The participant concerned.
    session : str
        The session concerned.
    kwargs : Any
        Any other information to store in the DataPoint.
    """

    image: tio.ScalarImage
    label: LabelType
    participant: str
    session: str

    def __init__(
        self,
        image: Union[tio.ScalarImage, PathType],
        label: Optional[Union[float, int, tio.LabelMap, PathType]],
        participant: str,
        session: str,
        **kwargs: Any,
    ) -> None:
        if isinstance(image, (Path, str)):
            image = tio.ScalarImage(path=image)

        if isinstance(label, (Path, str)):
            label = tio.LabelMap(path=label)

        super().__init__(
            image=image,
            label=label,
            participant=participant,
            session=session,
            **kwargs,
        )

    @property
    def shape(self):
        """
        Returns the shape of the images in the DataPoint.

        Consistency of shapes across images in the DataPoint is checked first.

        Examples
        --------
        >>> import torchio as tio
        >>> from clinicadl.data.structures import DataPoint
        >>> data = tio.datasets.Colin27()
        >>> datapoint = DataPoint(
                image=data.t1, label=data.brain, participant="sub-colin", session="ses-M000"
            )
        >>> datapoint.shape
        (1, 181, 217, 181)
        """
        return super().shape

    @property
    def spatial_shape(self):
        """
        Returns the spatial shape of the images in the DataPoint.

        Consistency of spatial shapes across images in the DataPoint is checked first.

        Examples
        --------
        >>> import torchio as tio
        >>> from clinicadl.data.structures import DataPoint
        >>> data = tio.datasets.Colin27()
        >>> datapoint = DataPoint(
                image=data.t1, label=data.brain, participant="sub-colin", session="ses-M000"
            )
        >>> datapoint.spatial_shape
        (181, 217, 181)
        """
        return super().spatial_shape

    @property
    def spacing(self):
        """
        Returns the voxel spacing of the images in the DataPoint.

        Consistency of voxel spacings across images in the DataPoint is checked first.

        Examples
        --------
        >>> import torchio as tio
        >>> from clinicadl.data.structures import DataPoint
        >>> data = tio.datasets.Colin27()
        >>> datapoint = DataPoint(
                image=data.t1, label=data.brain, participant="sub-colin", session="ses-M000"
            )
        >>> datapoint.spacing
        (1.0, 1.0, 1.0)
        """
        spacing = super().spacing
        return tuple(float(s) for s in spacing)

    @property
    def affine(self):
        """
        Returns affine matrix of the images in the DataPoint.

        Consistency of matrices across images in the DataPoint is checked first.

        Examples
        --------
        >>> import torchio as tio
        >>> from clinicadl.data.structures import DataPoint
        >>> data = tio.datasets.Colin27()
        >>> datapoint = DataPoint(
                image=data.t1, label=data.brain, participant="sub-colin", session="ses-M000"
            )
        >>> datapoint.affine
        array([[   1.,    0.,    0.,  -90.],
               [   0.,    1.,    0., -126.],
               [   0.,    0.,    1.,  -72.],
               [   0.,    0.,    0.,    1.]])
        """
        self.check_consistent_affine()
        return self.get_first_image().affine

    def add_image(
        self, image: Union[tio.ScalarImage, PathType], image_name: str
    ) -> None:
        """
        To add an image to the DataPoint.

        Parameters
        ----------
        image : Union[tio.ScalarImage, PathType]
            The image to add, as a :py:class:`torchio.ScalarImage` or a ``path`` to a NIfTI file.
        image_name : str
            The name that the image will take in the DataPoint.

        Examples
        --------
        >>> import torchio as tio
        >>> from clinicadl.data.structures import DataPoint
        >>> data = tio.datasets.Colin27()
        >>> datapoint = DataPoint(
                image=data.t1, label=data.brain, participant="sub-colin", session="ses-M000"
            )
        >>> datapoint.add_image(data.t1, "t1_bis")
        >>> datapoint["t1_bis"]
        ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)
        """
        if isinstance(image, (Path, str)):
            image = tio.ScalarImage(path=image)
        super().add_image(image, image_name)

    def add_mask(self, mask: Union[tio.LabelMap, PathType], mask_name: str) -> None:
        """
        To add a mask to the DataPoint.

        Parameters
        ----------
        mask : Union[tio.LabelMap, PathType]
            The mask to add, as a :py:class:`torchio.LabelMap` or a ``path`` to a NIfTI file.
        mask_name : str
            The name that the mask will take in the DataPoint.

        Examples
        --------
        >>> import torchio as tio
        >>> from clinicadl.data.structures import DataPoint
        >>> data = tio.datasets.Colin27()
        >>> datapoint = DataPoint(
                image=data.t1, label=data.brain, participant="sub-colin", session="ses-M000"
            )
        >>> datapoint.add_mask(data.head, "head")
        >>> datapoint["head"]
        LabelMap(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)
        """
        if isinstance(mask, (Path, str)):
            mask = tio.LabelMap(path=mask)
        super().add_image(mask, mask_name)

    def get_applied_transforms(
        self,
    ) -> list[tio.Transform]:
        """
        Gets the history of transforms applied to the DataPoint.

        Returns
        -------
        list[torchio.Transform]
            The history of transforms applied.
        """
        return super().get_applied_transforms()

    def plot(self, **kwargs) -> None:
        """
        Plots images using matplotlib.

        See :py:meth:`torchio.Subject.plot` for more details.
        """
        super().plot(**kwargs)

    def __copy__(self):
        return _subject_copy_helper(self, type(self))


def _subject_copy_helper(
    old_obj: tio.Subject,
    new_subj_cls: Callable[[Dict[str, Any]], tio.Subject],
):
    """
    Adapted from torchio.data.subject._subject_copy_helper to work
    with DataPoint.
    """
    result_dict = {}
    for key, value in old_obj.items():
        if isinstance(value, tio.Image):
            value = copy.copy(value)
        else:
            value = copy.deepcopy(value)
        result_dict[key] = value

    new = new_subj_cls(**result_dict)
    new.applied_transforms = old_obj.applied_transforms[:]
    return new


class Mask:
    """To handle masks in ClinicaDL. More precisely, it makes the difference
    between a mask passed as an image path, that corresponds to a common mask,
    and a mask passed as a suffix (a simple string), that corresponds to a mask
    specific to each image.

    For example, `Mask("masks/mask.nii.gz")` will be understood has a common
    mask, whereas `Mask("mask")` will be understood has an image-specific mask.

    In the latter case, it is expected that all the images studied
    have the associated mask in the CAPS directory.

    If the mask is in a `.pt` file (e.g. `Mask("masks/mask.pt")`), it is expected
    to be a 4D tensor with the associated affine matrix, as saved by
    `clinicadl.TensorConversion.save_mask_as_tensor`.\n
    If the mask is in a NIfTI file (e.g. `Mask("masks/mask.nii.gz")`), it is expected
    to be a 3D image.

    Parameters
    ----------
    mask : mask
        the mask, passed as a path or a suffix.

    Raises
    ------
    FileNotFoundError
        if `mask` is passed as a path that does not match any file.
    """

    def __init__(self, mask: PathType) -> None:
        if isinstance(mask, Path):
            if not self._check_path(mask):
                raise FileNotFoundError(
                    f"The mask has been passed as a Path object (got {mask}), but no such file exists."
                )
            self.is_common_mask = True
            self.path = Path(mask)
            self.name = self.path.with_suffix(
                ""
            ).stem  # with_suffix to handle double extensions

        elif isinstance(mask, str):
            if self._check_path(mask):
                self.is_common_mask = True
                self.path = Path(mask)
                self.name = self.path.with_suffix("").stem
            else:
                self.is_common_mask = False
                self.path = None
                self.name = mask

        self._mask_img: Union[tio.LabelMap, None] = None  # lazy loading

    @staticmethod
    def _check_path(mask_path: PathType) -> bool:
        """Checks if the mask file exists."""
        mask_path = Path(mask_path)
        return mask_path.is_file()

    def __str__(self):
        if self.is_common_mask:
            return f"Mask('{self.path}')"
        else:
            return f"Mask('{self.name}')"

    @classmethod
    def _load_mask(cls, path: Path) -> tio.LabelMap:
        """
        Loads a mask (in nifti or .pt file) and return a TorchIO LabelMap.
        """
        if path.suffix == PT:
            mask_tensor, affine = cls._load_pt_mask(path)
            return tio.LabelMap(tensor=mask_tensor, affine=affine)
        else:
            return tio.LabelMap(path=path)

    @staticmethod
    def _load_pt_mask(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a mask and its affine matrix from a .pt file.
        See also: :py:func:`clinicadl.data.tensor_conversion.TensorConversion.save_mask_as_tensor`.
        """
        pt_mask = torch.load(path, weights_only=True)
        return pt_mask[MASK], pt_mask[AFFINE]

    def _lazy_load_common_mask(self) -> tio.LabelMap:
        """
        Gets or loads a common mask (in nifti or .pt file).
        """
        if self._mask_img is None:
            self._mask_img = self._load_mask(self.path)
        return self._mask_img

    def _get_associated_mask_path(self, filename: Path) -> Path:
        """
        Returns the path of the mask associated to an image, when the
        mask is not a common mask.

        Examples
        --------
        >>> mask=Mask("brain")
        >>> mask._get_associated_mask_path("sub-000_ses-M000_pet.nii.gz")
        sub-000_ses-M000_brain.nii.gz
        """
        suffix = str(filename.with_suffix("").stem).rsplit("_", maxsplit=1)[
            -1
        ]  # with_suffix to handle double extensions
        mask_file = str(filename).replace(f"_{suffix}.", f"_{self.name}.")

        return Path(mask_file)

    def get_associated_mask(self, filename: Optional[PathType] = None) -> tio.LabelMap:
        """
        Returns the mask associated to an image, in a TorchIO LabelMap.

        If the mask is common to all subjects and sessions, the method will
        simply return it. On the other hand, if the mask is specific to each
        image, the method will use the input `filename` to get
        the associated mask.

        Parameters
        ----------
        filename : Optional[PathType], (optional, default=None)
            the image whose associated mask is to be found.
            Can be None if the mask is a common mask (thus it does not depend
            on 'filename').

        Returns
        -------
        tio.LabelMap :
            the mask, in a TorchIO LabelMap.

        Raises
        ------
        FileNotFoundError
            if the associated mask doesn't exist.

        Examples
        --------
        >>> mask=Mask("seg")
        >>> mask.get_associated_mask_path("sub-001_ses-M000_T1w.nii.gz")
        # will get the image in 'sub-001_ses-M000_seg.nii.gz'

        >>> mask=Mask("masks/leftHippocampus.nii.gz")
        >>> mask.get_associated_mask_path("sub-001_ses-M000_T1w.nii.gz")
        # will get the image in 'masks/leftHippocampus.nii.gz'
        >>> mask.get_associated_mask_path()
        # will get the image in 'masks/leftHippocampus.nii.gz'
        """
        if self.is_common_mask:
            return self._lazy_load_common_mask()
        else:
            if filename is None:
                raise ValueError(
                    f"The mask {self.name} is an image-specific mask, "
                    "you must therefore give a 'filename' to get the associated "
                    "mask."
                )

            filename = Path(filename)
            mask_file = self._get_associated_mask_path(filename)
            try:
                return self._load_mask(mask_file)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"No file matches {self.name}, so it is understood as a suffix. "
                    f"Therefore, the mask associated to {str(filename)} was expected "
                    f"to be found in {str(mask_file)}, but there is no such file."
                ) from exc
