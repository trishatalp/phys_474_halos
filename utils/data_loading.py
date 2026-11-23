"""Read the SPARC catalog, bulge luminosities, and radial measurements.

Return fresh tables with sample cuts and stellar masses for the analysis notebooks.
"""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from models.masses import stellar_mass


RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
GALAXY_PROPERTIES_PATH = RAW_DATA_DIR / "T1SPARC_Lelli2016c.mrt"
ROTATION_CURVES_PATH = RAW_DATA_DIR / "T2MassModels_Lelli2016c.mrt"
BULGE_LUMINOSITIES_PATH = RAW_DATA_DIR / "Bulges.mrt"


def _read_mrt_data(path: str | Path) -> StringIO:
    """Return the data after the final MRT header separator."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    separators = [i for i, line in enumerate(lines) if line.startswith("-----")]
    if not separators:
        raise ValueError(f"No MRT header separator found in {path}")
    return StringIO("\n".join(lines[separators[-1] + 1 :]))


GALAXY_CATALOG_COLUMNS = [
    "galaxy", "hubble_type", "distance_mpc", "distance_error_mpc", "distance_method",
    "inclination", "inclination_error", "total_luminosity_1e9_lsun",
    "total_luminosity_error_1e9_lsun", "effective_radius_kpc",
    "effective_surface_brightness", "disk_scale_length_kpc",
    "disk_central_surface_brightness", "hi_mass_1e9_msun", "hi_radius_kpc",
    "flat_velocity_kms", "flat_velocity_error_kms", "quality_flag", "references",
]
BULGE_COLUMNS = ["galaxy", "bulge_luminosity_1e9_lsun"]


def load_galaxy_catalog(path: str | Path = GALAXY_PROPERTIES_PATH) -> pd.DataFrame:
    """Read all SPARC Table 1 fields in their original units."""
    if Path(path).suffix.lower() == ".csv":
        return pd.read_csv(path, float_precision="round_trip")
    # The bundled file is padded differently from its documented byte ranges.
    # Split only the numeric fields so the final reference field stays intact.
    rows = [line.split(maxsplit=18) for line in _read_mrt_data(path) if line.strip()]
    catalog = pd.DataFrame(rows, columns=GALAXY_CATALOG_COLUMNS)
    for column in GALAXY_CATALOG_COLUMNS[1:-1]:
        catalog[column] = pd.to_numeric(catalog[column])
    return catalog


def load_bulge_catalog(path: str | Path = BULGE_LUMINOSITIES_PATH) -> pd.DataFrame:
    """Read the full bulge catalog in its original units."""
    if Path(path).suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep=r"\s+", comment="#", names=BULGE_COLUMNS)


def load_bulge_luminosities(path: str | Path = BULGE_LUMINOSITIES_PATH) -> pd.DataFrame:
    """Read SPARC's non-parametric bulge luminosities in solar luminosities.

    Source values are in 10^9 Lsun; outer bulge extrapolation introduces
    decomposition-dependent uncertainty. Explicit zero values are retained.
    """
    bulges = load_bulge_catalog(path).rename(
        columns={"bulge_luminosity_1e9_lsun": "bulge_luminosity_lsun"}
    )
    if bulges.galaxy.isna().any() or bulges.galaxy.duplicated().any():
        raise ValueError("Bulge table must contain unique, nonmissing galaxy names")
    luminosity = bulges["bulge_luminosity_lsun"]
    if not np.isfinite(luminosity).all() or (luminosity < 0).any():
        raise ValueError("Bulge luminosities must be finite and nonnegative")
    bulges["bulge_luminosity_lsun"] *= 1e9
    return bulges


def load_galaxy_properties(
    path: str | Path = GALAXY_PROPERTIES_PATH,
    *,
    bulge_luminosities_path: str | Path = BULGE_LUMINOSITIES_PATH,
    inclination_min: float = 30,
    inclination_max: float = 80,
) -> pd.DataFrame:
    """Load the cleaned galaxy sample from SPARC Table 1.

    Retain galaxy name, inclination, inclination_error (both in degrees),
    quality_flag, total luminosity and its error, joined bulge luminosity,
    and estimated stellar mass in solar masses.
    All luminosities are in solar luminosities at 3.6 microns. Exclude quality
    flag 3 and require the entire inclination uncertainty interval to lie
    within the inclusive bounds. Defaults select 108 galaxies.
    """
    if not 0 <= inclination_min <= inclination_max <= 90:
        raise ValueError("Inclination bounds must satisfy 0 <= min <= max <= 90")

    galaxy_properties = load_galaxy_catalog(path).rename(columns={
        "total_luminosity_1e9_lsun": "total_luminosity_lsun",
        "total_luminosity_error_1e9_lsun": "total_luminosity_error_lsun",
    })[[
        "galaxy", "inclination", "inclination_error",
        "total_luminosity_lsun", "total_luminosity_error_lsun", "quality_flag"
    ]]
    galaxy_properties[["total_luminosity_lsun", "total_luminosity_error_lsun"]] *= 1e9
    galaxy_properties = galaxy_properties.merge(
        load_bulge_luminosities(bulge_luminosities_path), on="galaxy",
        how="left", validate="one_to_one",
    )
    invalid = (
        galaxy_properties["bulge_luminosity_lsun"].isna()
        | (galaxy_properties["bulge_luminosity_lsun"] > galaxy_properties["total_luminosity_lsun"])
    )
    if invalid.any():
        names = galaxy_properties.loc[invalid, "galaxy"].tolist()
        raise ValueError(f"Missing bulge luminosity or bulge exceeds total for: {names}")
    selected = (
        (galaxy_properties["quality_flag"] != 3)
        & (
            galaxy_properties["inclination"]
            - galaxy_properties["inclination_error"]
            >= inclination_min
        )
        & (
            galaxy_properties["inclination"]
            + galaxy_properties["inclination_error"]
            <= inclination_max
        )
    )
    galaxy_properties = galaxy_properties.loc[selected].reset_index(drop=True)
    galaxy_properties["stellar_mass_msun"] = stellar_mass(
        galaxy_properties["total_luminosity_lsun"],
        galaxy_properties["bulge_luminosity_lsun"],
    )
    return galaxy_properties


def load_rotation_curves(path: str | Path = ROTATION_CURVES_PATH) -> pd.DataFrame:
    """Load SPARC Table 2 with the notebook's normalized column names.

    This source table retains all galaxies; sample cuts are applied when
    joining to the cleaned galaxy properties. Distance is in Mpc, radius in
    kpc, velocities in km/s, and surface brightness in solar luminosities/pcÂ².
    Signed gas velocities are preserved.
    """
    if Path(path).suffix.lower() == ".csv":
        return pd.read_csv(path, float_precision="round_trip")
    return pd.read_csv(
        _read_mrt_data(path),
        sep=r"\s+",
        names=[
            "galaxy", "distance", "radius", "v_obs", "v_err",
            "v_gas", "v_disk", "v_bulge", "sb_disk", "sb_bulge",
        ],
    )


def load_joined_rotation_curves(
    galaxy_properties_path: str | Path = GALAXY_PROPERTIES_PATH,
    rotation_curves_path: str | Path = ROTATION_CURVES_PATH,
    *,
    bulge_luminosities_path: str | Path = BULGE_LUMINOSITIES_PATH,
    inclination_min: float = 30,
    inclination_max: float = 80,
) -> pd.DataFrame:
    """Join rotation measurements to the cleaned galaxy properties.

    Return one row per radius measurement for selected galaxies, validating
    that each galaxy has only one properties row. Defaults yield 2,313 rows
    across 108 galaxies for the bundled SPARC files.
    """
    galaxy_properties = load_galaxy_properties(
        galaxy_properties_path,
        bulge_luminosities_path=bulge_luminosities_path,
        inclination_min=inclination_min,
        inclination_max=inclination_max,
    )
    rotation_curves = load_rotation_curves(rotation_curves_path)
    return rotation_curves.merge(
        galaxy_properties,
        on="galaxy",
        how="inner",
        validate="many_to_one",
    )
