"""Evaluate the enclosed mass and circular speed of an NFW halo.

The concentration is R200 divided by the scale radius. The enclosed mass at R200 is M200.
"""

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
    # At R200, x equals concentration, so this ratio gives unit enclosed mass fraction.
    return m200 * _shape(x) / _shape(concentration)

def velocity(radius, m200, concentration, g=G, h0=H0):
    mass = enclosed_mass(radius, m200, concentration, g, h0)
    return circular_velocity(radius, mass, g)
