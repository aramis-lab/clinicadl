from typing import List

import click

from clinicadl.utils import cli_param


@click.command(name="adapt", no_args_is_help=True)
@cli_param.argument.input_dir
@cli_param.argument.output_dir
@click.option(
    "--labels_list",
    "-lb",
    type = str,
    multiple = True,
    help="Labels used to create the tsv directory in the old way",
)
# @click.option(
#     "--split_type",
#     "-vt",
#     help="Type of split wanted for the validation: split or kfold",
#     default="kfold",
#     type=click.Choice(["split", "kfold"]),
# )
def cli(
    input_dir,
    output_dir, 
    labels_list = None
):

    from .adapt import adapt


    adapt(
        old_tsv_dir = input_dir,
        new_tsv_dir = output_dir,
        labels_list = labels_list
    )


if __name__ == "__main__":
    cli()
