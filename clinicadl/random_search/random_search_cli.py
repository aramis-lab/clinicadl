import click


@click.command("random-search", no_args_is_help=True)
@click.argument(
    "task",
    type=click.Choice(["classification", "vae-architecture"])
)
@click.argument(
    "launch_directory",
    type=str,
)
@click.argument(
    "toml_name", 
    type=str,
)
@click.argument("name", type=str)
def cli(
    task,
    launch_directory,
    name,
    toml_name,
):
    """Hyperparameter exploration using random search.

    LAUNCH_DIRECTORY is the path to the parents folder where results of random search will be saved.

    NAME is the name of the output folder containing the experiment.
    """

    if task == "classification":
        from .random_search import launch_search
        launch_search(task, launch_directory, name, toml_name)
    elif task == "vae-architecture":
        from.random_search import launch_vae_search
        launch_vae_search(launch_directory, name, toml_name)


if __name__ == "__main__":
    cli()
