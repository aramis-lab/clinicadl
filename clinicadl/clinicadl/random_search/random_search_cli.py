import click

from clinicadl.utils import cli_param


@click.command("random-search")
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
def cli(
    launch_directory,
    name,
    gpu,
    n_proc,
    batch_size,
    evaluation_steps,
):
    """Hyperparameter exploration using random search.

    LAUNCH_DIRECTORY is the path to the parents folder where results of random search will be saved.

    NAME is the name of the output folder containing the experiment.
    """
    from clinicadl.utils.cmdline_utils import check_gpu

    if gpu:
        check_gpu()

    from .random_search import launch_search

    options = {
        "batch_size": batch_size,
        "evaluation_steps": evaluation_steps,
        "n_proc": n_proc,
        "use_cpu": not gpu,
    }

    launch_search(launch_directory, name, options)


if __name__ == "__main__":
    cli()
