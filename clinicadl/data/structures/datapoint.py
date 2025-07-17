import copy
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import torchio as tio

from clinicadl.utils.typing import PathType

from .label import LabelType


class DataPoint(tio.Subject):
    """
    Data structure that gathers an image, the associated label, and any other relevant information
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

    Besides, a ``DataPoint`` is dictionary-like object. So, you can easily add a key-value pair
    to it:

    .. code-block:: python

        >>> datapoint["age"] = 55
        >>> datapoint["age"]    # the attribute notation won't work here
        55

    However, to add an image or a mask to the ``DataPoint``, prefer :py:func:`~add_image`
    and :py:func:`~add_mask`.

    To get all the images in your DataPoint, you can use :py:func:`get_images` or :py:func:`get_images_dict`.

    If all the images and masks of your DataPoint have the same shape, voxel spacing and affine matrix, you can easily
    access them via the attributes :py:attr:`~shape` (or :py:attr:`~spatial_shape` to remove the channel dimension),
    :py:attr:`~spacing` and :py:attr:`~affine` respectively.

    Finally,  you may also be interested in :py:func:`~plot` to plot images inside your ``DataPoint``, and :py:func:`~get_applied_transforms`
    to see the transforms applied to your data.

    As ``DataPoint`` is a subclass of :py:class:`torchio.Subject`, you can also used all the other methods it inherits from.

    .. note::
        Any transform used in ClinicaDL must work with DataPoint.

    Parameters
    ----------
    image : Union[torchio.ScalarImage, PathType]
        The image, as a :py:class:`torchio.ScalarImage` or a ``path`` to a NIfTI file.
    label : Optional[Union[float, int, dict[str, float], tio.LabelMap, PathType]]
        The label associated to the image. Can be:

        - a ``float`` (regression);
        - a ``dictionary`` with ``strings`` for keys and ``floats`` for values (multi-output regression);
        - an ``int`` (classification, including multi-class classification),
        - a mask, passed as a :py:class:`torchio.LabelMap` or a ``path`` to a NIfTI file, (segmentation);
        - or ``None``, if no label (reconstruction).

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
        label: Optional[Union[float, int, dict[str, float], tio.LabelMap, PathType]],
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
        Returns the shape of the images in the ``DataPoint``.

        Consistency of shapes across images in the ``DataPoint`` is checked first.

        Examples
        --------
        >>> from clinicadl.data.structures.examples import ColinDataPoint
        >>> datapoint = ColinDataPoint()
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
        >>> from clinicadl.data.structures.examples import ColinDataPoint
        >>> datapoint = ColinDataPoint()
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
        >>> from clinicadl.data.structures.examples import ColinDataPoint
        >>> datapoint = ColinDataPoint()
        >>> datapoint.spacing
        (1.0, 1.0, 1.0)
        """
        self.check_consistent_attribute("spacing", relative_tolerance=1e-3)
        return tuple(float(s) for s in self.image.spacing)

    @property
    def affine(self):
        """
        Returns affine matrix of the images in the ``DataPoint``.

        Consistency of matrices across images in the ``DataPoint`` is checked first
        (1e-3 relative tolerance).

        Examples
        --------
        >>> from clinicadl.data.structures.examples import ColinDataPoint
        >>> datapoint = ColinDataPoint()
        >>> datapoint.affine
        array([[   1.,    0.,    0.,  -90.],
               [   0.,    1.,    0., -126.],
               [   0.,    0.,    1.,  -72.],
               [   0.,    0.,    0.,    1.]])
        """
        self.check_consistent_attribute("affine", relative_tolerance=1e-3)
        return self.image.affine

    def get_images(
        self,
        intensity_only=True,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> list[tio.Image]:
        """
        To get the list of all the images in a ``DataPoint``.

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
        list[torchio.Image]
            The list of the :py:class:`torchio.Image`.

        Examples
        --------
        >>> from clinicadl.data.structures import ColinDataPoint
        >>> datapoint = ColinDataPoint()
        >>> datapoint.get_images()
        [ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)]
        >>> datapoint.get_images(intensity_only=False)
        [ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...),
        LabelMap(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...),
        LabelMap(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)]

        See Also
        --------
        :py:meth:`~DataPoint.get_images_dict`
        """
        return super().get_images(intensity_only, include, exclude)

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
        >>> from clinicadl.data.structures import ColinDataPoint
        >>> datapoint = ColinDataPoint()
        >>> datapoint.get_images_dict()
        {'image': ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)}

        See Also
        --------
        :py:meth:`~DataPoint.get_images`
        """
        return super().get_images_dict(intensity_only, include, exclude)

    def add_image(
        self, image: Union[tio.ScalarImage, PathType], image_name: str
    ) -> None:
        """
        To add an image to the ``DataPoint``.

        Parameters
        ----------
        image : Union[tio.ScalarImage, PathType]
            The image to add, as a :py:class:`torchio.ScalarImage` or a ``path`` to a NIfTI file.
        image_name : str
            The name that the image will take in the DataPoint.

        Examples
        --------
        >>> from clinicadl.data.structures import ColinDataPoint
        >>> datapoint = ColinDataPoint()
        >>> datapoint
        ColinDataPoint(Keys: ('image', 'label', 'participant', 'session', 'head'); images: 3)
        >>> datapoint.add_image(datapoint.image, "image_duplicate")
        >>> datapoint
        ColinDataPoint(Keys: ('image', 'label', 'participant', 'session', 'head', 'image_duplicate'); images: 4)
        >>> datapoint["image_duplicate"]
        ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)

        See Also
        --------
        :py:meth:`~DataPoint.add_mask`
        """
        if isinstance(image, (Path, str)):
            image = tio.ScalarImage(path=image)
        super().add_image(image, image_name)

    def add_mask(self, mask: Union[tio.LabelMap, PathType], mask_name: str) -> None:
        """
        To add a mask to the ``DataPoint``.

        Parameters
        ----------
        mask : Union[tio.LabelMap, PathType]
            The mask to add, as a :py:class:`torchio.LabelMap` or a ``path`` to a NIfTI file.
        mask_name : str
            The name that the mask will take in the ``DataPoint``.

        Examples
        --------
        >>> from clinicadl.data.structures import ColinDataPoint
        >>> datapoint = ColinDataPoint()
        >>> datapoint
        ColinDataPoint(Keys: ('image', 'label', 'participant', 'session', 'head'); images: 3)
        >>> datapoint.add_mask(datapoint["head"], "head_duplicate")
        >>> datapoint
        ColinDataPoint(Keys: ('image', 'label', 'participant', 'session', 'head', 'head_duplicate'); images: 4)
        >>> datapoint["head_duplicate"]
        LabelMap(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; path: ...)

        See Also
        --------
        :py:meth:`~DataPoint.add_image`
        """
        if isinstance(mask, (Path, str)):
            mask = tio.LabelMap(path=mask)
        super().add_image(mask, mask_name)

    def get_applied_transforms(
        self,
    ) -> list[tio.Transform]:
        """
        Gets the history of transforms applied to the ``DataPoint``.

        Returns
        -------
        list[torchio.Transform]
            The history of transforms applied.

        Examples
        --------
        >>> from clinicadl.data.structures import ColinDataPoint
        >>> from torchio import RescaleIntensity
        >>> datapoint = ColinDataPoint()
        >>> transform = RescaleIntensity()
        >>> datapoint = transform(datapoint)
        >>> datapoint.get_applied_transforms()
        [RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), masking_method=None, in_min_max=(0.0, 9646287.0))]
        """
        return super().get_applied_transforms()

    def plot(self, **kwargs) -> None:
        """
        Plots images using matplotlib.

        See :py:meth:`torchio.Subject.plot` for more details.
        """
        super().plot(**kwargs)

    def __copy__(self):
        return _subject_copy_helper(self, DataPoint)


def _subject_copy_helper(
    old_obj: DataPoint,
    new_subj_cls: Callable[[Dict[str, Any]], DataPoint],
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
