"""NFW halo; concentration is R200 / scale radius."""
import numpy as np
from .constants import G, H0
from .virial import r200, validate_radius, circular_velocity

def _shape(x):
    x = np.asarray(x)
    # Avoid subtractive cancellation near the origin.
    series = x*x * (0.5 + x*(-2/3 + x*(3/4 - 4*x/5)))
    return np.where(x < 1e-4, series, np.log1p(x) - x/(1+x))

def enclosed_mass(radius, m200, concentration, g=G, h0=H0):
    radius = validate_radius(radius, concentration)
    x = radius * concentration / r200(m200, g, h0)
    return m200 * _shape(x) / _shape(concentration)

def velocity(radius, m200, concentration, g=G, h0=H0):
    radius = validate_radius(radius, concentration)
    return circular_velocity(radius, enclosed_mass(radius, m200, concentration, g, h0), g)
