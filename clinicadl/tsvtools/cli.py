import click

from .analysis.analysis_cli import cli as analysis_cli
from .getlabels.getlabels_cli import cli as getlabels_cli
from .kfold.kfold_cli import cli as kfold_cli
from .split.split_cli import cli as split_cli


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(cls=RegistrationOrderGroup, name="tsvtools", no_args_is_help=True)
def cli() -> None:
    """
    Manipulation of TSV files to prepare and manage input data.
    """
    pass


cli.add_command(getlabels_cli)
cli.add_command(analysis_cli)
cli.add_command(split_cli)
cli.add_command(kfold_cli)

if __name__ == "__main__":
    cli()
