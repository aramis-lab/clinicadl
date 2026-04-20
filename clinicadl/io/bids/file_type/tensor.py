from typing import Iterable, Optional, Pattern

from typing_extensions import Self

from .base import AlphanumericStr, BidsFileType

CONVERSION = "conv"


class Tensor(BidsFileType):
    """
    A :py:class:`~clinicadl.io.BidsFileType` to represent tensor files.

    Parameters
    ----------
    conversion_name : str
        The name of the tensor conversion associated to these tensors. The name can be
        found in the filenames associated with the key ``"conv"``.
    entities : Optional[dict[AlphanumericStr, Pattern]], default=None
        The entities that are in the filenames of the tensors.
    """

    def __init__(
        self,
        conversion_name: str,
        entities: Optional[dict[AlphanumericStr, Pattern]] = None,
    ):
        entities = entities or {}
        entities[CONVERSION] = conversion_name
        super().__init__(
            data_type="tensors",
            suffix="tensors",
            extension=".pt",
            with_entities=entities,
            description=f"Outputs of the tensor conversion '{conversion_name}'.",
        )

    @classmethod
    def from_source_file_types(
        cls, conversion_name: str, file_types: Iterable[BidsFileType]
    ) -> Self:
        """
        To create a ``Tensor`` object from the :py:class:`BidsFileTypes <clinicadl.io.BidsFileType>`
        corresponding to the data that are inside the tensor files.

        A tensor file can contain heterogeneous data (e.g., an image and a mask). Here, the entities
        in the tensor filenames are inferred from these source data.

        Parameters
        ----------
        conversion_name : str
            The name of the tensor conversion associated to these tensors.
        entities : Iterable[BidsFileType]
            The ``BidsFileTypes`` from which the tensor conversion has been performed.
        """
        common_entities = _entities_intersection(file_types)

        return cls(conversion_name, common_entities)


def _entities_intersection(file_types: Iterable[BidsFileType]) -> dict[str, str]:
    """
    Gets all the common entities to the input BidsFileType.
    """
    inter_keys = set.intersection(
        *[set(file_type.with_entities.keys()) for file_type in file_types]
    )
    common_entities = {}
    for key in inter_keys:
        if (
            unique := _unique_alphanum(
                [file_type.with_entities[key].pattern for file_type in file_types]
            )
        ) is not None:
            common_entities[key] = unique

    return common_entities


def _unique_alphanum(values: Iterable[str]) -> str | None:
    """
    If there is a unique alphanumeric string in the iterable, it is returned.
    """
    unique_values: set[str] = set(values)
    if len(unique_values) > 1:
        return None

    if (unique := unique_values.pop()).isalnum():
        return unique
