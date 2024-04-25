import click

from clinicadl.generate.generate_config import SharedGenerateConfigOne

config = SharedGenerateConfigOne.model_fields

caps_directory = click.argument(
    "caps_directory", type=config["caps_directory"].annotation
)
generated_caps_directory = click.argument(
    "generated_caps_directory", type=config["generated_caps_directory"].annotation
)
