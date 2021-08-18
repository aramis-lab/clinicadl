import click

from .generate_random_cli import cli as generate_random_cli
from .generate_shepplogan_cli import cli as generate_shepplogan_cli
from .generate_trivial_cli import cli as generate_trivial_cli


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(cls=RegistrationOrderGroup, name="generate")
def cli() -> None:
    """Generation of synthetic dataset."""
    pass


cli.add_command(generate_random_cli)
cli.add_command(generate_trivial_cli)
cli.add_command(generate_shepplogan_cli)


if __name__ == "__main__":
    cli()
