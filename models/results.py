"""Serializable fit configuration and result tables. No file I/O."""
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd
from .constants import G, H0, Y_DISK, Y_BULGE

@dataclass(frozen=True)
class FitConfig:
    g: float = G
    h0: float = H0
    y_disk: float = Y_DISK
    y_bulge: float = Y_BULGE
    bounds: tuple | None = None
    factors: tuple = (0.5, 1., 2.)
    method: str = "L-BFGS-B"
    optimizer_options: dict = field(default_factory=dict)
    agreement_delta_chi2: float = 0.01
    bound_tolerance: float = 1e-5

    def __post_init__(self):
        if not np.isfinite([self.g, self.h0, self.y_disk, self.y_bulge]).all() or min(self.g, self.h0) <= 0 or min(self.y_disk, self.y_bulge) < 0:
            raise ValueError("Invalid constants or mass-to-light ratios")
        if self.bounds is None:
            # Li et al. (2020), section 2.3, page 4. Convert V200 to log10(M200/Msun).
            mass_limits = tuple(float(np.log10(v**3 / (10 * self.g * self.h0)))
                                for v in (10.0, 500.0))
            # The halo formulas require positive concentration.
            object.__setattr__(self, "bounds", (mass_limits, (0.001, 1000.0)))
        b = np.asarray(self.bounds)
        if b.shape != (2, 2) or not np.isfinite(b).all() or np.any(b[:, 0] >= b[:, 1]) or b[1, 0] <= 0:
            raise ValueError("Invalid parameter bounds")
        if not self.factors or not np.isfinite(self.factors).all() or min(self.factors) <= 0:
            raise ValueError("Initialization factors must be positive")
        if self.method != "L-BFGS-B":
            raise ValueError("Only L-BFGS-B is supported")
        if self.agreement_delta_chi2 <= 0 or self.bound_tolerance < 0:
            raise ValueError("Invalid diagnostic tolerances")

    def to_dict(self):
        return asdict(self)

@dataclass
class GalaxyFitResult:
    summary: dict
    attempts: list = field(default_factory=list)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)

PREDICTION_COLUMNS = ["galaxy", "model", "radius", "v_obs", "v_err", "v_bar_squared", "v_halo", "v_model", "residual", "normalized_residual"]
ATTEMPT_COLUMNS = ["galaxy", "model", "start", "initial_log_m200", "initial_concentration", "log_m200", "concentration", "success", "optimizer_status", "message", "chi2", "iterations", "evaluated_at_bound"]

def result_tables(results):
    """Convert a sequence of GalaxyFitResult records to three DataFrames."""
    summary = pd.DataFrame([r.summary for r in results])
    attempts = pd.DataFrame([a for r in results for a in r.attempts], columns=ATTEMPT_COLUMNS)
    predictions = [r.predictions for r in results if not r.predictions.empty]
    predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=PREDICTION_COLUMNS)
    return summary, attempts, predictions
