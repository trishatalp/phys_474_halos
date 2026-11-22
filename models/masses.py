"""Stellar masses from SPARC 3.6-micron luminosities, in solar units."""
import numpy as np
from .constants import Y_DISK, Y_BULGE


def stellar_mass(total_luminosity_lsun, bulge_luminosity_lsun,
                 y_disk=Y_DISK, y_bulge=Y_BULGE):
    """Return total stellar mass in Msun using separate disk/bulge ratios.

    The disk luminosity is total minus bulge. This estimate inherits the
    SPARC decomposition assumptions and adopts fixed mass-to-light ratios;
    it does not include gas, dark matter, or an uncertainty estimate.
    """
    total, bulge = np.broadcast_arrays(
        np.asarray(total_luminosity_lsun, dtype=float),
        np.asarray(bulge_luminosity_lsun, dtype=float),
    )
    if not np.isfinite([y_disk, y_bulge]).all() or min(y_disk, y_bulge) < 0:
        raise ValueError("Mass-to-light ratios must be finite and nonnegative")
    if (not np.isfinite(total).all() or not np.isfinite(bulge).all()
            or np.any(total < 0) or np.any(bulge < 0) or np.any(bulge > total)):
        raise ValueError("Luminosities must be finite and satisfy 0 <= bulge <= total")
    return y_disk * (total - bulge) + y_bulge * bulge
