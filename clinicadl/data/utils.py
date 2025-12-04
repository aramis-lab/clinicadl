"""
Other functions to perform various utility task.
"""

import shutil

from clinicadl.utils.typing import PathType

from .tensors import TensorConversionInfo


def remove_tensors(json_path: PathType) -> None:
    """
    To delete tensors in a dataset.

    Will remove all the tensors, saved with :py:class:`clinicadl.data.datasets.CapsDataset.to_tensors` for example,
    associated to ``conversion_name``, as well as the associated ``JSON`` file in ``<your-dataset>/tensor_conversion``.

    Parameters
    ----------
    json_path : PathType
        Path to the ``json`` file associated to the tensor conversion you want to delete.

    Examples
    --------

    .. code-block::

        from clinicadl.data import datasets, datatypes
        from clinicadl.data.utils import remove_tensors

        caps_dataset = datasets.CapsDataset(
            caps_directory="my_caps", preprocessing=datatypes.T1Linear(use_uncropped_image=True)
        )
        caps_dataset.to_tensors()  # the json file will be "my_caps/tensor_conversion/default_t1-linear.json"

        remove_tensors("my_caps/tensor_conversion/default_t1-linear.json")

    See Also
    --------
    :py:class:`clinicadl.data.datasets.CapsDataset.to_tensors`
    """
    base_dir = json_path.parents[1]

    tensor_conversion = TensorConversionInfo.from_json(json_path)

    for path in base_dir.rglob(str(tensor_conversion.tensors_location)):
        shutil.rmtree(path)

    json_path.unlink()
