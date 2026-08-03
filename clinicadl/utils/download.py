from pathlib import Path

from platformdirs import user_cache_dir


def get_clinicadl_cache_dir() -> Path:
    """Return the default cache directory for ClinicaDL data."""
    return Path(user_cache_dir("clinicadl"))
