import click

from .adapt.adapt_cli import cli as adapt_cli
from .analysis.analysis_cli import cli as analysis_cli
from .get_labels.get_labels_cli import cli as get_labels_cli
from .get_labels.old_get_labels_cli import cli as old_get_labels_cli
from .get_metadata.get_metadata_cli import cli as get_metadata_cli
from .get_progression.get_progression_cli import cli as get_progression_cli
from .kfold.kfold_cli import cli as kfold_cli
from .prepare_experiment.prepare_experiment_cli import cli as prepare_experiment_cli
from .split.split_cli import cli as split_cli

# class RegistrationOrderGroup(click.Group):
#     """CLI group which lists commands by order or registration."""

#     def list_commands(self, ctx):
#         return self.commands.keys()


@click.group(name="tsvtools", no_args_is_help=True)
def cli() -> None:
    """
    Manipulation of TSV files to prepare and manage input data.
    """
    pass


cli.add_command(get_labels_cli)
cli.add_command(old_get_labels_cli)
cli.add_command(analysis_cli)
cli.add_command(split_cli)
cli.add_command(kfold_cli)
cli.add_command(prepare_experiment_cli)
cli.add_command(get_metadata_cli)
cli.add_command(get_progression_cli)
cli.add_command(adapt_cli)

if __name__ == "__main__":
    cli()
