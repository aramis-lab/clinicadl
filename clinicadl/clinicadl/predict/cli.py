import click

from .predict_cli import cli as infer_cli


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()

@click.group(cls=RegistrationOrderGroup, name="infer")
def cli() -> None:
    """Descrption"""
    pass

cli.add_command(infer_cli)

if __name__ == "__main__":
    cli()