class BidsEntity(str):
    """
    A :bids:`BIDS entity <common-principles.html#entities>`.

    Examples
    --------
    .. code-block::
        >>> bids = BidsEntity("trc-18FFDG")
        >>> bids.key
        'trc'
        >>> bids.value
        '18FFDG'
    """

    def __init__(self, entity: str):
        if "-" not in entity:
            raise ValueError(
                f"A BIDS entity must be of the form '<key>-<value>'. Got '{entity}'"
            )
        self.key, _, self.value = entity.partition("-")
        assert self.key.isalnum(), f"They key of a BIDS entity must be an alphanumeric string. Got: '{self.key}'"
        assert self.value.isalnum(), f"They value of a BIDS entity must be an alphanumeric string or an int. Got: '{self.value}'"


class Subject(BidsEntity):
    """
    A :py:class:`BidsEntity` representing a :bids:`subject <appendices/entities.html#sub>` (a.k.a. a
    "participant").
    """

    def __init__(self, entity: str):
        super().__init__(entity)
        assert (
            self.key == "sub"
        ), f"A participant id must start with 'sub' (e.g., 'sub-001'). Got '{self.key}'"


class Session(BidsEntity):
    """
    A :py:class:`BidsEntity` representing a :bids:`session <appendices/entities.html#ses>`.
    """

    def __init__(self, entity: str):
        super().__init__(entity)
        assert (
            self.key == "ses"
        ), f"A session id must start with 'ses' (e.g., 'ses-M000'). Got '{self.key}'"
