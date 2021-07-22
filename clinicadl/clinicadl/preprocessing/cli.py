import click


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()

@click.group(name="preprocessing")
def cli() -> None:
    """Descrption"""
    pass

cli.add_command()

if __name__ == "__main__":
    cli()
