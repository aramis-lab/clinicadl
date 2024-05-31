import click

from clinicadl.generate.generate_config import GenerateHypometabolicConfig
from clinicadl.utils.enum import Pathology

config_hypometabolic = GenerateHypometabolicConfig.model_fields
pathology = click.option(
    "--pathology",
    "-p",
    type=click.Choice(Pathology),
    default=config_hypometabolic["pathology"].default.value,
    help="Pathology applied. To chose in the following list: [ad, bvftd, lvppa, nfvppa, pca, svppa]",
    show_default=True,
)
anomaly_degree = click.option(
    "--anomaly_degree",
    "-anod",
    type=config_hypometabolic["anomaly_degree"].annotation,
    default=config_hypometabolic["anomaly_degree"].default,
    help="Degrees of hypo-metabolism applied (in percent)",
    show_default=True,
)
sigma = click.option(
    "--sigma",
    type=config_hypometabolic["sigma"].annotation,
    default=config_hypometabolic["sigma"].default,
    help="It is the parameter of the gaussian filter used for smoothing.",
    show_default=True,
)
