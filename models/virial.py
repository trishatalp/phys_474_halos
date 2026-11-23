"""Convert between halo mass, virial radius, and circular speed.

R200 encloses a sphere whose mean density is 200 times the critical density.
"""

import numpy as np
from .constants import G, H0

def positive(value, name):
    if not np.all(np.isfinite(value)) or np.any(np.asarray(value) <= 0):
        raise ValueError(f"{name} must be finite and positive")

def critical_density(g=G, h0=H0):
    positive(g, "G")
    positive(h0, "H0")
    return 3 * h0**2 / (8 * np.pi * g)

def r200(m200, g=G, h0=H0):
    positive(m200, "M200")
    return (3 * np.asarray(m200) / (4 * np.pi * 200 * critical_density(g, h0)))**(1/3)

def validate_radius(radius, concentration):
    positive(concentration, "concentration")
    radius = np.asarray(radius, dtype=float)
    if not np.all(np.isfinite(radius)) or np.any(radius < 0):
        raise ValueError("radius must be finite and nonnegative")
    return radius

def circular_velocity(radius, mass, g=G):
    radius, mass = np.broadcast_arrays(radius, mass)
    # Both profiles have zero circular speed at the origin.
    speed_squared = np.zeros_like(radius, dtype=float)
    np.divide(g * mass, radius, out=speed_squared, where=radius > 0)
    return np.sqrt(speed_squared)
