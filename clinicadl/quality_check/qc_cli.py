import click

from .t1_linear.cli import cli as t1_linear_cli
from .t1_volume.cli import cli as t1_volume_cli


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(cls=RegistrationOrderGroup, name="quality-check")
def cli() -> None:
    """Performs quality check procedure for t1-linear or t1-volume pipelines.

    Original code can be found at https://github.com/vfonov/deep-qc
    """
    pass


cli.add_command(t1_linear_cli)
cli.add_command(t1_volume_cli)

if __name__ == "__main__":
    cli()
