import click

from clinicadl.generate.generate_config import GenerateHypometabolicConfig
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

pathology = click.option(
    "--pathology",
    "-p",
    type=get_type("pathology", GenerateHypometabolicConfig),
    default=get_default("pathology", GenerateHypometabolicConfig),
    help="Pathology applied. To chose in the following list: [ad, bvftd, lvppa, nfvppa, pca, svppa]",
    show_default=True,
)
anomaly_degree = click.option(
    "--anomaly_degree",
    "-anod",
    type=get_type("anomaly_degree", GenerateHypometabolicConfig),
    default=get_default("anomaly_degree", GenerateHypometabolicConfig),
    help="Degrees of hypo-metabolism applied (in percent)",
    show_default=True,
)
sigma = click.option(
    "--sigma",
    type=get_type("sigma", GenerateHypometabolicConfig),
    default=get_default("sigma", GenerateHypometabolicConfig),
    help="It is the parameter of the gaussian filter used for smoothing.",
    show_default=True,
)
