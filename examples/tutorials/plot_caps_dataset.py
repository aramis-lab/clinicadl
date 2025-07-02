"""
Building a basic CapsDataset
============================

This example shows how to build a :py:class:`~clinicadl.data.datasets.CapsDataset` to manipulate your data.
"""

# %%
# Create a CapsDataset from a CAPS directory
# ------------------------------------------

from pathlib import Path

from clinicadl.data import datasets, datatypes

current_dir = Path.cwd()
caps_path = current_dir.parent / "resources" / "caps"
data = caps_path / "data.tsv"
caps = datasets.CapsDataset(
    caps_path,
    data=data,
    preprocessing=datatypes.PETLinear(
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
