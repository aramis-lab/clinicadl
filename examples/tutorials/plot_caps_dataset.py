"""
Building a basic CapsDataset
============================

This example shows how to build a :py:class:`~clinicadl.data.datasets.CapsDataset` to manipulate your data.
"""

# %%
# Create a CapsDataset from a CAPS directory
# ------------------------------------------

from pathlib import Path

from clinicadl.data import datasets, datatypes, utils

current_dir = Path.cwd()
caps_path = current_dir.parent / "resources" / "caps"
data = caps_path / "data.tsv"
caps = datasets.CapsDataset(
    caps_path,
    data=data,
    datatype=datatypes.PETLinear(
        use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
    ),
)

# %%
# Convert your images to tensors
# ------------------------------

# This is a python comment
caps.to_tensors()


# %%
# Plot an image
# -------------
caps[0].plot()

# %%
# Get data on the subjects
# ------------------------
caps.df

# %%
# Tip: Remove tensors if you don't need them anymore
# --------------------------------------------------
#
# Since we didn't pass ``conversion_name`` to ``to_tensors``, a default conversion name
# was generated: ``"default_pet-linear_18FAV45_pons2"``
utils.remove_tensors(
    caps_path / "tensor_conversion" / "default_pet-linear_18FAV45_pons2.json"
)

# %%
