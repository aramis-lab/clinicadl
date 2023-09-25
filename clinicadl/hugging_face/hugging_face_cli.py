import click

from .pull_cli import cli as pull_cli
from .push_cli import cli as push_cli


@click.group(name="hugging-face", no_args_is_help=True)
def cli():
    """Train a deep learning model for a specific task."""
    pass


cli.add_command(push_cli)
cli.add_command(pull_cli)


if __name__ == "__main__":
    cli()
