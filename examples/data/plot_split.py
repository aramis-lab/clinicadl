"""
Split your data
===============

This example shows how to split your data into training, validation and test sets.
"""

# %%
# Load the (participant, session) couples
# ---------------------------------------
#
# ClinicaDL makes splits from a list of (participant, session).

from pathlib import Path

import pandas as pd

caps_path = (
    Path("../../") / "tests" / "unittests" / "resources" / "caps_example"
).resolve()
participants_sessions = caps_path / "tsv" / "labels.tsv"
pd.read_csv(participants_sessions, sep="\t").head(5)

# %%
# Make a train/test split
# -----------------------
#
# First, let's keep some (participant, session) in an independent test set.
# To perform a simple split, we will use :py:func:`~clinicadl.splitter.make_split`.

from clinicadl.splitter import make_split

split_dir = make_split(
    participants_sessions,
    n_test=0.25,
    output_dir=Path("../tmp"),
    stratification=["age", "diagnosis"],
    p_categorical_threshold=0.3,
    p_continuous_threshold=0.3,
)

# %%
# The split is now stored in ``split_dir``. In particular, there is a file named
# ``train.tsv`` that contains the training set, and a file named ``test_baseline.py`` that contains
# the test set.

pd.read_csv(split_dir / "train.tsv", sep="\t").head(5)

# %%
pd.read_csv(split_dir / "test_baseline.tsv", sep="\t").head(5)

# %%
# .. note::
#   To have a robust estimation of your model performance, it is advised to test your model
#   on only one image for each participant. By default, :py:func:`~clinicadl.splitter.make_split`
#   will thus only put the baseline session (i.e. the first session) of the test patients in the
#   test set. This is why the file is named ``test_baseline``. If you want to test your model on
#   all the sessions of the test participants, put ``longitudinal=True`` in :py:func:`~clinicadl.splitter.make_split`.

# %%
# Make a K-Fold split for Cross-Validation
# ----------------------------------------
#
# Now that we have isolated our test set, we want to make a K-Fold split on the
# remaining data to perform cross-validation. To do this, we will use :py:func:`~clinicadl.splitter.make_kfold`.

from clinicadl.splitter import make_kfold

kfold_dir = make_kfold(
    split_dir
    / "train.tsv",  # here the input data is all the data that is not in the test set
    n_splits=2,
    longitudinal=True,
)

# %%
# We didn't specify ``output_dir``, so by default ``make_kfold`` will store the results in ``split_dir``.
# Let's have a look at the first split of our 2-Fold.

# %%
pd.read_csv(split_dir / "2_fold" / "split-0" / "train.tsv", sep="\t").head(5)

# %%
pd.read_csv(split_dir / "2_fold" / "split-0" / "validation.tsv", sep="\t").head(5)

# %%
# .. note::
#   Here, we chose to validate our model on all the sessions, and not only on the baseline sessions. That's why
#   we use ``validation.tsv`` and not ``validation_baseline.tsv``.

# %%
# Split a dataset
# ---------------
#
# Now that we have built our train, validation, and test groups, we will use these splits
# to split a :py:class:`~clinicadl.data.datasets.CapsDataset`.

# %%
# It is straightforward to get a test and a training dataset:

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import PETLinear

preprocessing = PETLinear(
    tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
)

train_set = CapsDataset(
    caps_path, preprocessing=preprocessing, data=split_dir / "train.tsv"
)
train_set.df

# %%
test_set = CapsDataset(
    caps_path, preprocessing=preprocessing, data=split_dir / "test_baseline.tsv"
)
test_set.df

# %%
# Regarding the validation sets, it would be heavy to do that for each split of the
# K-Fold. We will rather use :py:class:`~clinicadl.splitter.KFold`, that will handle the splits for us:

from clinicadl.splitter import KFold

splitter = KFold(kfold_dir)

# %%
# ``KFold`` reads the split directory. We can then split any ``CapsDataset``, using
# :py:class:`KFold.get_splits <clinicadl.splitter.KFold.get_splits>`. This method is a generator
# that enables to iterate over the splits of the K-Fold.

for i, split in enumerate(splitter.get_splits(train_set)):
    print(f"Split {i}")
    print(f"Training set: {len(split.train_dataset)} images")
    print(f"Test set: {len(split.val_dataset)} images")

# %%
# Manipulate a :py:class:`~clinicadl.splitter.Split`
# --------------------------------------------------
#
# :py:class:`~clinicadl.splitter.KFold.get_splits` returns :py:class:`~clinicadl.splitter.Split` objects.
# A ``Split`` contains the data of the training/validation splits, as well as other relevant information
# for ClinicaDL. Before passing it to the ``Trainer``, you will have to build the associated :py:class:`~torch.utils.data.DataLoader` s:

split.build_train_loader(batch_size=2, shuffle=True)
split.build_val_loader(batch_size=2, shuffle=False)
split.train_loader

# %%
#
# ----
#
# To remove the split directories:

import shutil

shutil.rmtree(split_dir)
