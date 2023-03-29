from typing import List

import click

from clinicadl.utils import cli_param


@click.command(name="adapt", no_args_is_help=True)
@cli_param.argument.old_tsv_dir
@cli_param.argument.new_tsv_dir
@click.option(
    "--labels_list",
    "-lb",
    type=str,
    multiple=True,
    help="Labels used to create the tsv directory in the old way",
)
def cli(old_tsv_dir, new_tsv_dir, labels_list=None):
    """Converts split and kfold directories created with clinicaDL 1.1.1 and earlier version
    to the last version.

    OLD_TSV_DIR is the output directory of the split/kfold pipeline that contains all the TSV files with clinicaDL 1.1.1 and earlier versions.

    Results are stored in NEW_TSV_DIR.
    """

    from clinicadl.utils.exceptions import ClinicaDLArgumentError

    from .adapt import adapt

    if new_tsv_dir.is_dir():
        raise ClinicaDLArgumentError(
            f"\nThe directory: {new_tsv_dir} already exists.\n"
            "Please give another path for the new tsv directory."
        )

    adapt(old_tsv_dir=old_tsv_dir, new_tsv_dir=new_tsv_dir, labels_list=labels_list)


if __name__ == "__main__":
    cli()
