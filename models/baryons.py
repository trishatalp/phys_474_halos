"""Compute the baryonic contribution to a rotation curve.

Combine signed gas terms with disk and bulge terms scaled by their mass-to-light ratios.
"""

import numpy as np
from .constants import Y_DISK, Y_BULGE

def components_squared(v_gas, v_disk, v_bulge, y_disk=Y_DISK, y_bulge=Y_BULGE):
    if not np.isfinite([y_disk, y_bulge]).all() or min(y_disk, y_bulge) < 0:
        raise ValueError("Mass-to-light ratios must be finite and nonnegative")

    # SPARC uses negative gas speeds for an outward contribution to the radial force.
    gas, disk, bulge = np.broadcast_arrays(v_gas, v_disk, v_bulge)
    return {"gas": gas * np.abs(gas), "disk": y_disk * disk**2, "bulge": y_bulge * bulge**2}

def velocity_squared(v_gas, v_disk, v_bulge, y_disk=Y_DISK, y_bulge=Y_BULGE):
    return sum(components_squared(v_gas, v_disk, v_bulge, y_disk, y_bulge).values())
