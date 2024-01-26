from pathlib import Path

from clinicadl.train.train_utils import bids_to_caps

bids_to_caps(
    Path("/Users/camille.brianceau/aramis/DATA/bids_QC_OASIS1"),
    Path("/Users/camille.brianceau/aramis/DATA/caps_QC_OASIS1_bis"),
)
