from collections import OrderedDict
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict

from clinicadl.dictionary.words import NAME


class ClinicaDLConfig(BaseModel):
    """Base configuration class."""

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Customized version of 'model_dump'.

        Returns the serialized config class.
        """
        return _order_dict(self.model_dump())


def _order_dict(model_or_field: Any) -> Any:
    """
    To always have the field 'name' at the beginning.

    Recursive function to handle fields that themeselves
    contain 'ClinicaDLConfig' instances.
    """
    if isinstance(model_or_field, dict):
        ordered_dict = OrderedDict(**model_or_field)
        if NAME in ordered_dict:  # always 'name' at the beginning
            ordered_dict.move_to_end(NAME, last=False)

        for key, value in ordered_dict.items():
            ordered_dict[key] = _order_dict(value)

        return ordered_dict

    elif isinstance(model_or_field, (tuple, list)):
        ordered_sequence = []
        for v in model_or_field:
            ordered_sequence.append(_order_dict(v))
        if isinstance(model_or_field, tuple):
            ordered_sequence = tuple(ordered_sequence)

        return ordered_sequence

    return model_or_field
