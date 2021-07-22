import click

from .prepare_data_cli import cli as prepare_data_cli


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(cls=RegistrationOrderGroup, name="extract-tensor")
def cli() -> None:
    """
    Extraction of pytorch tensor from nifty images.
    """
    pass


cli.add_command(prepare_data_cli)


if __name__ == "__main__":
    cli()
