"""Burkert halo normalized to M200, with concentration = R200 / r0."""
import numpy as np
from .constants import G, H0
from .virial import r200, validate_radius, circular_velocity

def _shape(x):
    x = np.asarray(x)
    series = 4*x**3/3 - x**4 + 4*x**7/7 - x**8/2
    return np.where(x < 1e-3, series, 2*np.log1p(x) + np.log1p(x*x) - 2*np.arctan(x))

def enclosed_mass(radius, m200, concentration, g=G, h0=H0):
    radius = validate_radius(radius, concentration)
    x = radius * concentration / r200(m200, g, h0)
    return m200 * _shape(x) / _shape(concentration)

def velocity(radius, m200, concentration, g=G, h0=H0):
    radius = validate_radius(radius, concentration)
    return circular_velocity(radius, enclosed_mass(radius, m200, concentration, g, h0), g)
