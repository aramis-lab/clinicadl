import click

from clinicadl.utils import cli_param

cmd_name = "restrict"


@click.command(name=cmd_name)
@cli_param.argument.dataset
@cli_param.argument.merged_tsv
@click.argument(
    "results_tsv",
    type=str,
)
def cli(dataset, merged_tsv, results_tsv):
    """Reproduce restrictions applied to AIBL and OASIS datasets.

    DATASET is either AIBL or OASIS.

    MERGED_TSV is the path to the file obtained by the command `clinica iotools merge-tsv`.

    Results are stored in RESULTS_TSV.
    """
    from .restrict import aibl_restriction, oasis_restriction

    if dataset == "AIBL":
        aibl_restriction(merged_tsv, results_tsv)
    elif dataset == "OASIS":
        oasis_restriction(merged_tsv, results_tsv)


if __name__ == "__main__":
    cli()
