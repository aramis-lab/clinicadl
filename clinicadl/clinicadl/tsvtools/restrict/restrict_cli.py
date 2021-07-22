import click

from clinicadl.utils import cli_param

cmd_name = "restrict"


@click.command(name=cmd_name)
@cli_param.argument.dataset
@cli_param.argument.merged_tsv
@cli_param.argument.results_directory
def cli(dataset, merged_tsv, results_directory):
    """
    """
    # import function to execute
    from .restrict import aibl_restriction, oasis_restriction
    # run function
    if dataset == "AIBL":
        aibl_restriction(merged_tsv, results_directory)
    elif dataset == "OASIS":
        oasis_restriction(merged_tsv, results_directory)
    

if __name__ == "__main__":
    cli()
