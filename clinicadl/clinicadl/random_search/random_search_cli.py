import click

from clinicadl.utils import cli_param


@click.command(name="generate")
@click.argument(
    "launch_directory",
    type=str,
)
@click.argument("name", type=str)
@cli_param.option.use_gpu
@cli_param.option.n_proc
@cli_param.option.batch_size
@click.option(
    "--evaluation_steps",
    "-esteps",
    default=0,
    type=int,
    help="""Fix the number of iterations to perform before computing an evaluation. Default will only
    perform one evaluation at the end of each epoch.""",
)
def rs_generate_cli(
    launch_directory,
    name,
    use_gpu,
    n_proc,
    batch_size,
    evaluation_steps,
):
    """Hyperparameter exploration using random search.

    LAUNCH_DIRECTORY is the path to the parents folder where results of random search will be saved.

    NAME is the name of the output folder containing the experiment.
    """
    from .random_search import launch_search

    options = {
        "batch_size": batch_size,
        "evaluation_steps": evaluation_steps,
        "n_proc": n_proc,
        "use_cpu": not use_gpu,
    }

    launch_search(launch_directory, name, options)


@click.command(name="analysis")
@click.argument(
    "launch_directory",
    type=str,
)
def rs_analysis_cli(launch_directory):
    """Performs the analysis of all jobs in random search directory.

    LAUNCH_DIRECTORY is the path to the parents folder where al results of random search are saved.
    """
    from clinicadl.utils.meta_maps.random_search_analysis import random_search_analysis

    random_search_analysis(launch_directory)


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(cls=RegistrationOrderGroup, name="random-search")
def cli() -> None:
    """Generate random networks to explore hyper parameters space."""
    pass


cli.add_command(rs_generate_cli)
cli.add_command(rs_analysis_cli)

if __name__ == "__main__":
    cli()
