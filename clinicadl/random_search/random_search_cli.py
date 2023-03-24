from pathlib import Path

import click


@click.command("random-search", no_args_is_help=True)
@click.argument(
    "launch_directory",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument("name", type=str)
def cli(
    launch_directory,
    name,
):
    """Hyperparameter exploration using random search.

    LAUNCH_DIRECTORY is the path to the parents folder where results of random search will be saved.

    NAME is the name of the output folder containing the experiment.
    """
    from .random_search import launch_search

    launch_search(launch_directory, name)


if __name__ == "__main__":
    cli()
