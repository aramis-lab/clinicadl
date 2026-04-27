from typing import Iterable, Optional, Pattern

from typing_extensions import Self

from .base import AlphanumericStr, BidsFile, BidsFileType

CONVERSION = "conv"


class TensorType(BidsFileType):
    """
    A :py:class:`~clinicadl.io.BidsFileType` to represent tensor files.

    Parameters
    ----------
    entities : Optional[dict[AlphanumericStr, Pattern]], default=None
        The entities that are in the filenames of the tensors.
    """

    def __init__(
        self,
        entities: Optional[dict[AlphanumericStr, Pattern]] = None,
    ):
        entities = entities or {}
        super().__init__(
            data_type="tensors",
            suffix="tensors",
            extension=".pt",
            with_entities=entities,
            description="Outputs of the tensor conversion.",
        )

    @classmethod
    def from_source_file_types(
        cls,
        conversion_name: str,
        image: BidsFileType,
        individual_masks: Iterable[BidsFileType],
        common_masks: Iterable[BidsFile],
        transformed: bool,
    ) -> Self:
        """
        To create a ``TensorType`` object from the :py:class:`BidsFileTypes <clinicadl.io.BidsFileType>`
        and :py:class:`BidsFile <clinicadl.io.BidsFile>` corresponding to the data that are inside the tensor files.

        A tensor file can contain heterogeneous data (e.g., an image and a mask). Here, the entities
        in the tensor filenames are inferred from these source data.

        Parameters
        ----------
        conversion_name : str
            The name of the tensor conversion associated to these tensors.
        image : BidsFileType
            The ``BidsFileType`` associated to the image.
        individual_masks : Iterable[BidsFileType]
            The ``BidsFileTypes`` associated to the image-specific masks.
        common_masks : Iterable[BidsFile]
            The ``BidsFiles`` associated to the non-image-specific masks.
        transformed : bool
            If the data were transformed during the conversion.
        """
        if transformed:
            entities = {}
        else:
            entities = _entities_intersection(
                [{key: value.pattern for key, value in image.with_entities.items()}]
                + [
                    {
                        key: value.pattern
                        for key, value in file_type.with_entities.items()
                    }
                    for file_type in individual_masks
                ]
                + [file.entities for file in common_masks]
            )
        if (suffix := image.suffix.pattern).isalnum():
            entities["src"] = suffix

        entities[CONVERSION] = conversion_name

        return cls(entities)


def _entities_intersection(entities: Iterable[dict[str, str]]) -> dict[str, str]:
    """
    Gets all the common entities from a list of dict of entities.
    """
    inter_keys = set.intersection(*[set(e.keys()) for e in entities])
    common_entities = {}
    for key in inter_keys:
        print(entities)
        if (unique := _unique_alphanum([e[key] for e in entities])) is not None:
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
