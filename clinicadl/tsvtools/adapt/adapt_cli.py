from typing import List

import click

from clinicadl.utils import cli_param


@click.command(name="adapt", no_args_is_help=True)
@cli_param.argument.input_dir
@cli_param.argument.output_dir
@click.option(
    "--labels_list",
    "-lb",
    type=str,
    multiple=True,
    help="Labels used to create the tsv directory in the old way",
)
def cli(input_dir, output_dir, labels_list=None):
    """Converts split and kfold directories created with clinicaDL 1.1.1 and earlier version
    to the last version.

    INPUT_DIR is the output of th split/kfold directory of the old version.

    Results are stored in OUTPUT_DIRECTORY.
    """

    import os

    from clinicadl.utils.exceptions import ClinicaDLArgumentError

    from .adapt import adapt

    if os.path.exists(output_dir):
        raise ClinicaDLArgumentError(
            f"\nThe directory: {output_dir} already exists.\n"
            "Please give another path for the new tsv directory."
        )

    adapt(old_tsv_dir=input_dir, new_tsv_dir=output_dir, labels_list=labels_list)


if __name__ == "__main__":
    cli()
