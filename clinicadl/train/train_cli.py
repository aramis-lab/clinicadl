import click

from .from_json_cli import cli as from_json_cli
from .list_models_cli import cli as list_models_cli
from .resume_cli import cli as resume_cli
from .tasks.classification_cli import cli as classification_cli
from .tasks.reconstruction_cli import cli as reconstruction_cli
from .tasks.regression_cli import cli as regression_cli


@click.group(name="train", no_args_is_help=True)
def cli():
    """Train a deep learning model for a specific task."""
    pass


cli.add_command(classification_cli)
cli.add_command(regression_cli)
cli.add_command(reconstruction_cli)
cli.add_command(from_json_cli)
cli.add_command(resume_cli)
cli.add_command(list_models_cli)


if __name__ == "__main__":
    cli()
