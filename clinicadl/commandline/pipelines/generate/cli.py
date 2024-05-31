import click

from clinicadl.commandline.pipelines.generate.artifacts.cli import (
    cli as generate_artifacts_cli,
)
from clinicadl.commandline.pipelines.generate.hypometabolic.cli import (
    cli as generate_hypo_cli,
)
from clinicadl.commandline.pipelines.generate.random.cli import (
    cli as generate_random_cli,
)
from clinicadl.commandline.pipelines.generate.shepplogan.cli import (
    cli as generate_shepplogan_cli,
)
from clinicadl.commandline.pipelines.generate.trivial.cli import (
    cli as generate_trivial_cli,
)


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(cls=RegistrationOrderGroup, name="generate", no_args_is_help=True)
def cli() -> None:
    """Generation of synthetic dataset."""
    pass


cli.add_command(generate_random_cli)
cli.add_command(generate_trivial_cli)
cli.add_command(generate_shepplogan_cli)
cli.add_command(generate_hypo_cli)
cli.add_command(generate_artifacts_cli)


if __name__ == "__main__":
    cli()
