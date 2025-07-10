from pathlib import Path
from typing import Union

from clinicadl.dictionary.suffixes import JSON
from clinicadl.dictionary.words import DEFAULT

from .readers import CapsReader
from .tensor_conversion import TensorConversionInfo


def remove_tensors(caps_directory: Union[str, Path], conversion_name: str) -> None:
    """
    To remove some tensors of a :term:`CAPS` directory.

    Will remove all the tensors saved with :py:class:`clinicadl.data.datasets.CapsDataset.to_tensors`
    and associated to ``conversion_name``, as well as the associated ``JSON`` file in ``{caps_directory}/tensor_conversion``.

    Parameters
    ----------
    caps_directory : Union[str, Path]
        The :term:`CAPS` directory where to remove the tensors.
    conversion_name : str
        The name of the ``JSON`` file ``{caps_directory}/tensor_conversion``, without the ``.json``
        extension.

    Examples
    --------

    .. code-block::

        from clinicadl.data import datasets, datatypes
        from clinicadl.data.utils import remove_tensors

        caps_dataset = datasets.CapsDataset(
            caps_directory="my_caps", preprocessing=datatypes.T1Linear(use_uncropped_image=True)
        )
        caps_dataset.to_tensors()  # the json file will be "default_t1-linear.json"

        remove_tensors(caps_directory="my_caps", conversion_name="default_t1-linear")

    .. code-block::

        caps_dataset.to_tensors(conversion_name="a_conversion")

        remove_tensors(caps_directory="my_caps", conversion_name="a_conversion")

    See Also
    --------
    :py:class:`clinicadl.data.datasets.CapsDataset.to_tensors`
    """
    caps_reader = CapsReader(Path(caps_directory))

    if conversion_name.startswith(DEFAULT):
        tensor_folder_name = DEFAULT
    else:
        tensor_folder_name = conversion_name

    json_name = Path(conversion_name).with_suffix(JSON)
    json = caps_reader.tensor_conversion_json_dir / json_name

    tensor_conversion = TensorConversionInfo.from_json(json)

    for participant, session in tensor_conversion.participants_sessions:
        pt_path = caps_reader.get_tensor_path(
            participant,
            session,
            tensor_conversion.preprocessing,
            conversion_name=tensor_folder_name,
        )
        pt_path.unlink()

        is_empty = not any(pt_path.parent.iterdir())
        if is_empty:
            pt_path.parent.rmdir()

    json.unlink()
