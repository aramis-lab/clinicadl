import click

from .classification.cli import cli as classification_cli
from .from_json.cli import cli as from_json_cli
from .list_models.cli import cli as list_models_cli
from .reconstruction.cli import cli as reconstruction_cli
from .regression.cli import cli as regression_cli
from .resume.cli import cli as resume_cli


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
