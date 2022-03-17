from .mmd import MaximumMeanDiscrepancy
from .dsbn import DSBN1d, DSBN2d
from .mixup import mixup
from .efdmix import (
    EFDMix, random_efdmix, activate_efdmix, run_with_efdmix, deactivate_efdmix,
    crossdomain_efdmix, run_without_efdmix
)
from .mixstyle import (
    MixStyle, random_mixstyle, activate_mixstyle, run_with_mixstyle,
    deactivate_mixstyle, crossdomain_mixstyle, run_without_mixstyle
)
from .transnorm import TransNorm1d, TransNorm2d
from .sequential2 import Sequential2
from .reverse_grad import ReverseGrad
from .cross_entropy import cross_entropy
from .optimal_transport import SinkhornDivergence, MinibatchEnergyDistance
