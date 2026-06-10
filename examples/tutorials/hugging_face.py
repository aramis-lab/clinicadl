"""
Using HuggingFace with ClinicaDL
================================

Here’s the workflow for using ClinicaDL with Hugging Face Hub
"""

# %%
# Install dependencies
# ------------------------------------------
#
# Besides `clinicadl`, you’ll need the Hugging Face Hub library:

# pip install huggingface_hub


# %%
# Authenticate with Hugging Face
# ------------------------------
#
# Run this once on your machine:

# huggingface-cli login

# Paste your token (you can create one at Hugging Face Settings → Access Tokens)


# %%
# Save and push your ClinicaDL model
# ----------------------------------
#
# Suppose your trained model is in `my_model`

from huggingface_hub import HfApi, Repository

org = "your-organization"
username = "your-username"
repo_name = "clinicadl-model"

repo_id = f"{org}/{repo_name}"  # or f"{username}/{repo_name}" for user accounts

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=False)

repo = Repository(local_dir="my_model", clone_from=repo_id)

# Commit and push
repo.git_add()
repo.git_commit("Initial commit: add ClinicaDL model")
repo.git_push()

# ..note::
# It’s good practice to add a `README.md` or `model_card.md` describing your model, so the
# Hugging Face Hub page looks nice.

# %%
