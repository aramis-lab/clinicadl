from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
import torchio as tio
from pydantic import field_validator
from torch import Tensor

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.typing import PathType
from clinicadl.utils.variables import SPACING_RTOL


class DataPointConfig(ClinicaDLConfig):
    """To check ``DataPoint`` inputs."""

    image: tio.ScalarImage
    participant_id: str
    session_id: str

    @field_validator("image", mode="before")
    @classmethod
    def _validate_image(cls, value: Any) -> Any:
        """Loads the image if it is a path."""
        if isinstance(value, (Path, str)):
            return tio.ScalarImage(path=value)
        return value


class DataPoint(tio.Subject):
    """
    Data structure that gathers an image and any other relevant information
    associated to the image.

    It inherits from :py:class:`torchio.Subject`, which inherits itself from Python's ``dict``.

    A ``DataPoint`` can contain any type of values, but you are encouraged to store your images
    in :py:class:`torchio.ScalarImage` and your masks in :py:class:`torchio.LabelMap`.

    A DataPoint has the following attributes:
        - ``image``: the image, in a :py:class:`torchio.ScalarImage`;
        - ``participant_id``: the id of the participant, in a ``str``;
        - ``session_id``: the id of the session, in a ``str``.

    You can easily access these elements using the attribute notation:

    .. code-block:: python

        >>> import torchio as tio
        >>> import torch
        >>> import numpy as np
        >>> datapoint = DataPoint(
                image=tio.ScalarImage(tensor=torch.randn(1, 10, 10, 10), affine=np.eye(4)),
                participant_id="sub-001",
                session_id="ses-M000",
            )
        >>> datapoint.session_id
        'ses-M000'

    To add, modify, or delete any other field, you can use the standard dictionary syntax:

    .. code-block:: python

        >>> datapoint["age"] = 55
        >>> datapoint["age"]
        55

    To add an image or a mask to the ``DataPoint``, prefer :py:func:`~add_image`
    and :py:func:`~add_mask`.

    To get all the images in your DataPoint, you can use :py:func:`get_images_dict`.

    If all the images and masks of your DataPoint have the same shape, voxel spacing or affine matrix, you can easily
    access them via the attributes :py:attr:`~shape` (or :py:attr:`~spatial_shape` to remove the channel dimension),
    :py:attr:`~spacing` and :py:attr:`~affine` respectively.

    Finally,  you may also be interested in :py:func:`~plot` to plot images inside your ``DataPoint``.

    As ``DataPoint`` is a subclass of :py:class:`torchio.Subject`, you can also used all the other methods it inherits from.

    Parameters
    ----------
    image : Union[torchio.ScalarImage, PathType]
        The image, as a :py:class:`torchio.ScalarImage` or a ``path`` to a NIfTI file.
    participant_id : str
        The participant id.
    session_id : str
        The session id.
    kwargs : Any
        Any other information to store in the ``DataPoint``.
    """

    image: tio.ScalarImage
    participant_id: str
    session_id: str

    def __init__(
        self,
        image: Union[tio.ScalarImage, PathType],
        participant_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> None:
        config = DataPointConfig(
            image=image,
            participant_id=participant_id,
            session_id=session_id,
        )
        kwargs.update(config.to_raw_dict())

        super().__init__(**kwargs)

    @property
    def shape(self):
        """
        Returns the shape of the images in the ``DataPoint``.

        Consistency of shapes across images in the ``DataPoint`` is checked first.

        Examples
        --------
        >>> from clinicadl.data.structures.examples import Colin27DataPoint
        >>> datapoint = Colin27DataPoint()
        >>> datapoint.shape
        (1, 181, 217, 181)
        """
        return super().shape

    @property
    def spatial_shape(self):
        """
        Returns the spatial shape of the images in the ``DataPoint``.

        Consistency of spatial shapes across images in the ``DataPoint`` is checked first.

        Examples
        --------
        >>> from clinicadl.data.structures.examples import Colin27DataPoint
        >>> datapoint = Colin27DataPoint()
        >>> datapoint.spatial_shape
        (181, 217, 181)
        """
        self.check_consistent_attribute("spatial_shape")
        return self.get_first_image().spatial_shape

    @property
    def spacing(self):
        """
        Returns the voxel spacing of the images in the ``DataPoint``.

        Consistency of voxel spacings across images in the ``DataPoint`` is checked first
        (1e-3 relative tolerance).

        Examples
        --------
        >>> from clinicadl.data.structures.examples import Colin27DataPoint
        >>> datapoint = Colin27DataPoint()
        >>> datapoint.spacing
        (1.0, 1.0, 1.0)
        """
        self.check_consistent_attribute("spacing", relative_tolerance=SPACING_RTOL)
        return tuple(float(s) for s in self.image.spacing)

    @property
    def affine(self):
        """
        Returns affine matrix of the images in the ``DataPoint``.

        Consistency of matrices across images in the ``DataPoint`` is checked first
        (1e-3 relative tolerance).

        Examples
        --------
        >>> from clinicadl.data.structures.examples import Colin27DataPoint
        >>> datapoint = Colin27DataPoint()
        >>> datapoint.affine
        array([[   1.,    0.,    0.,  -90.],
               [   0.,    1.,    0., -126.],
               [   0.,    0.,    1.,  -72.],
               [   0.,    0.,    0.,    1.]])
        """
        self.check_consistent_attribute("affine", relative_tolerance=1e-3)
        return self.image.affine

    def get_images_dict(
        self,
        intensity_only: bool = True,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> dict[str, tio.Image]:
        """
        To get all the images in a ``DataPoint``, and their names.

        Parameters
        ----------
        intensity_only : bool, default=True
            To get only the images (:py:class:`torchio.ScalarImage`) and not the
            masks (:py:class:`torchio.LabelMap`).
        include : Optional[Sequence[str]], default=None
            Names of the images to include. If ``None``, will return all the images
            specified by ``intensity_only`` and not in ``exclude``.
        exclude : Optional[Sequence[str]], default=None
            Names of the images to exclude.

        Returns
        -------
        dict[str, torchio.Image]
            The images and their names.

        Examples
        --------
        >>> from clinicadl.data.structures.examples import Colin27DataPoint
        >>> datapoint = Colin27DataPoint()
        >>> datapoint.get_images_dict()
        {'image': ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)}

        See Also
        --------
        :py:meth:`~DataPoint.get_masks_dict`
        """
        return super().get_images_dict(intensity_only, include, exclude)

    def get_masks_dict(
        self,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> dict[str, tio.LabelMap]:
        """
        To get all the masks in a ``DataPoint``, and their names.

        Parameters
        ----------
        include : Optional[Sequence[str]], default=None
            Names of the masks to include. If ``None``, will return all the masks
            not in ``exclude``.
        exclude : Optional[Sequence[str]], default=None
            Names of the masks to exclude.

        Returns
        -------
        dict[str, torchio.LabelMap]
            The masks and their names.

        Examples
        --------
        >>> from clinicadl.data.structures.examples import Colin27DataPoint
        >>> datapoint = Colin27DataPoint()
        >>> datapoint.get_masks_dict()
        {'head': LabelMap(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)}

        See Also
        --------
        :py:meth:`~DataPoint.get_images_dict`
        """
        images = super().get_images_dict(intensity_only=True).keys()

        return {
            name: mask
            for name, mask in self.get_images_dict(
                intensity_only=False, include=include, exclude=exclude
            ).items()
            if name not in images
        }

    def get_non_images_dict(
        self,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        """
        To get all the values in the ``DataPoint`` that are not images or masks.

        Parameters
        ----------
        include : Optional[Sequence[str]], default=None
            Keys to include. If ``None``, will return all the keys
            not in ``exclude``.
        exclude : Optional[Sequence[str]], default=None
            Keys to exclude.

        Returns
        -------
        dict[str, Any]
            The non-image values and their keys.

        Examples
        --------
        >>> from clinicadl.data.structures.examples import Colin27DataPoint
        >>> datapoint = Colin27DataPoint()
        >>> datapoint.get_non_images_dict()
        {'participant_id': 'sub-colin', 'session_id': 'ses-M000'}

        See Also
        --------
        :py:meth:`~DataPoint.get_images_dict`
        """
        keys = set(self.get_keys()) - set(self.get_images_names()) - set(exclude or {})
        if include is not None:
            keys = keys.intersection(include)

        return {name: value for name, value in self.items() if name in keys}

    def get_image_tensor(self, image_name: str) -> Tensor:
        """
        Returns a copy of the tensor associated to an image that is a :py:class:`torchio.Image`.

        Parameters
        ----------
        image_name : str
            The name of the image in the ``DataPoint``.

        Returns
        -------
        torch.Tensor
            The tensor image.
        """
        if not isinstance(field_value := self[image_name], tio.Image):
            raise TypeError(
                f"{image_name} is a {type(field_value)}, not a torchio.Image!"
            )

        return field_value.tensor.clone()

    def get_keys(
        self,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> list[str]:
        """
        To get the list of all the keys in a ``DataPoint``.

        Parameters
        ----------
        include : Optional[Sequence[str]], default=None
            Keys to include. If ``None``, will return all the keys not in ``exclude``.
        exclude : Optional[Sequence[str]], default=None
            Names of the keys to exclude.

        Returns
        -------
        list[str]
            The keys.

        Examples
        --------
        >>> from clinicadl.data.structures.examples import Colin27DataPoint
        >>> datapoint = Colin27DataPoint()
        >>> datapoint.get_keys()
        ['image', 'head', 'participant_id', 'session_id']
        >>> datapoint.get_keys(exclude=["image"])
        ['head', 'participant_id', 'session_id']
        >>> datapoint.get_keys(include=["image"])
        ['image']
        """
        keys = set(self.keys())
        if include is not None:
            keys = keys.intersection(include)
        if exclude is not None:
            keys -= set(exclude)

        return list(keys)

    def add_image(
        self,
        image: Union[tio.ScalarImage, PathType, torch.Tensor],
        image_name: str,
    ) -> None:
        """
        To add an image to the ``DataPoint``.

        Parameters
        ----------
        image : Union[tio.ScalarImage, PathType, torch.Tensor]
            The image to add, as a :py:class:`torchio.ScalarImage`, a path to the NIfTI file containing the image,
            or a 4D :py:class:`torch.Tensor` (including one channel dimension). If a ``Tensor`` is passed, the same affine matrix as ``self.image``
            will be used.
        image_name : str
            The name that the image will take in the ``DataPoint``.

        Examples
        --------
        >>> from clinicadl.data.structures.examples import Colin27DataPoint
        >>> datapoint = Colin27DataPoint()
        >>> datapoint
        Colin27DataPoint(Keys: ('head', 'image', 'participant_id', 'session_id'); images: 2)
        >>> datapoint.add_image(datapoint.image, "image_duplicate")
        >>> datapoint
        Colin27DataPoint(Keys: ('head', 'image', 'participant_id', 'session_id', 'image_duplicate'); images: 3)
        >>> datapoint["image_duplicate"]
        ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)

        See Also
        --------
        :py:meth:`~DataPoint.add_mask`
        """
        if isinstance(image, (Path, str)):
            image = tio.ScalarImage(path=image)
        elif isinstance(image, torch.Tensor):
            image = tio.ScalarImage(tensor=image, affine=self.image.affine)

        super().add_image(image, image_name)

    def add_mask(
        self, mask: Union[tio.ScalarImage, PathType, torch.Tensor], mask_name: str
    ) -> None:
        """
        To add a mask to the ``DataPoint``.

        Parameters
        ----------
        mask : Union[tio.ScalarImage, PathType, torch.Tensor]
            The mask to add, as a :py:class:`torchio.LabelMap`, a path to the NIfTI file containing the image,
            or a 4D :py:class:`torch.Tensor` (including one channel dimension). If a ``Tensor`` is passed, the same affine matrix as ``self.image```
            will be used.
        mask_name : str
            The name that the mask will take in the ``DataPoint``.

        Examples
        --------
        >>> from clinicadl.data.structures.examples import Colin27DataPoint
        >>> datapoint = Colin27DataPoint()
        >>> datapoint
        Colin27DataPoint(Keys: ('head', 'image', 'participant_id', 'session_id'); images: 2)
        >>> datapoint.add_mask(datapoint["head"], "head_duplicate")
        >>> datapoint
        Colin27DataPoint(Keys: ('head', 'image', 'participant_id', 'session_id', 'head_duplicate'); images: 3)
        >>> datapoint["head_duplicate"]
        LabelMap(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)

        See Also
        --------
        :py:meth:`~DataPoint.add_image`
        """
        if isinstance(mask, (Path, str)):
            mask = tio.LabelMap(path=mask)
        elif isinstance(mask, torch.Tensor):
            mask = tio.LabelMap(tensor=mask, affine=self.image.affine)

        super().add_image(mask, mask_name)

    def plot(self, **kwargs) -> None:
        """
        Plots images using matplotlib.

        See :py:meth:`torchio.Subject.plot` for more details.
        """
        super().plot(**kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.update_attributes()

    def __delitem__(self, key):
        super().__delitem__(key)
        if hasattr(self, key):
            delattr(self, key)

    def remove_image(self, image_name: str) -> None:
        """
        Removes an image from the ``DataPoint``.
        """
        self._check_image_name(image_name)
        del self[image_name]
