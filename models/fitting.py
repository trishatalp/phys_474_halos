"""Shared deterministic multistart fitting; no plotting or file I/O."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from . import nfw, burkert, baryons
from .results import FitConfig, GalaxyFitResult

HALOS = {"nfw": nfw, "burkert": burkert}

def initial_guesses(max_observed_velocity, config):
    """Legacy heuristic for BOTH halos; maximum Vobs is not measured Vflat."""
    if not np.isfinite(max_observed_velocity) or max_observed_velocity <= 0:
        raise ValueError("Maximum observed velocity must be positive")
    mass = max_observed_velocity**3 / (10 * config.g * config.h0)
    h = config.h0 * 10
    concentration = 10**(0.905 - 0.101*np.log10(mass*h/1e12))
    guesses = np.array([[np.log10(mass*m), concentration*c] for m in config.factors for c in config.factors])
    # Explicitly constrain starts, matching the bounded optimizer's behavior.
    bounds = np.asarray(config.bounds)
    return np.clip(guesses, bounds[:, 0], bounds[:, 1])

def total_velocity(v_bar_squared, v_halo):
    squared = np.asarray(v_bar_squared) + np.asarray(v_halo)**2
    if not np.isfinite(squared).all() or np.any(squared < 0):
        raise ValueError("Invalid total model velocity squared")
    return np.sqrt(squared)

def residuals(v_obs, v_model):
    return np.asarray(v_obs) - v_model

def chi_squared(params, radius, v_obs, v_err, v_bar_squared, model, config):
    halo = HALOS[model].velocity(radius, 10**params[0], params[1], config.g, config.h0)
    try:
        predicted = total_velocity(v_bar_squared, halo)
    except ValueError:
        return np.inf
    return float(np.sum((residuals(v_obs, predicted)/v_err)**2))

def bound_hits(params, config):
    return [bool(np.any(np.isclose(p, bounds, atol=config.bound_tolerance, rtol=0))) for p, bounds in zip(params, config.bounds)]

def select_best(attempts):
    successful = [a for a in attempts if a["success"] and np.isfinite(a["chi2"])]
    return min(successful, key=lambda a: a["chi2"]) if successful else None

def fit_galaxy(galaxy_data, model, config=None):
    config = config or FitConfig()
    if model not in HALOS:
        raise ValueError(f"Unknown halo model: {model}")
    name = str(galaxy_data["galaxy"].iloc[0]) if len(galaxy_data) and "galaxy" in galaxy_data else "unknown"
    summary = dict(galaxy=name, model=model, status="invalid_data", message="", n_points=len(galaxy_data),
                   log_m200=np.nan, m200=np.nan, concentration=np.nan, chi2=np.nan,
                   dof=len(galaxy_data)-2, reduced_chi2=np.nan, successful_starts=0, agreeing_starts=0,
                   mass_bound_hit=False, concentration_bound_hit=False)
    result = GalaxyFitResult(summary)
    try:
        if galaxy_data["galaxy"].isna().any() or galaxy_data["galaxy"].nunique() != 1:
            raise ValueError("Expected exactly one named galaxy")
        data = galaxy_data.sort_values("radius")
        values = data[["radius", "v_obs", "v_err", "v_gas", "v_disk", "v_bulge"]].to_numpy(dtype=float)
        if len(data) <= 2 or not np.isfinite(values).all():
            raise ValueError("Need more than two finite measurements")
        radius, observed, error, gas, disk, bulge = values.T
        if np.any(radius <= 0) or np.any(error <= 0):
            raise ValueError("Radii and velocity errors must be positive")
        bar = baryons.velocity_squared(gas, disk, bulge, config.y_disk, config.y_bulge)
        guesses = initial_guesses(observed.max(), config)
    except (ValueError, KeyError, TypeError) as exc:
        summary["message"] = str(exc)
        return result
    for index, guess in enumerate(guesses):
        attempt = dict(galaxy=name, model=model, start=index+1, initial_log_m200=guess[0],
                       initial_concentration=guess[1], log_m200=np.nan, concentration=np.nan,
                       success=False, optimizer_status=-1, message="", chi2=np.nan, iterations=0)
        attempt["evaluated_at_bound"] = False

        def tracked_objective(params, *args):
            if not attempt["evaluated_at_bound"]:
                attempt["evaluated_at_bound"] = any(bound_hits(params, config))
            return chi_squared(params, *args)

        try:
            fit = minimize(tracked_objective, guess, args=(radius, observed, error, bar, model, config),
                           method=config.method, bounds=config.bounds, options=config.optimizer_options)
            attempt.update(log_m200=float(fit.x[0]), concentration=float(fit.x[1]),
                           success=bool(fit.success and np.isfinite(fit.fun) and np.isfinite(fit.x).all()),
                           optimizer_status=int(fit.status), message=str(fit.message),
                           chi2=float(fit.fun), iterations=int(fit.nit))
        except (ValueError, FloatingPointError, OverflowError) as exc:
            attempt["message"] = str(exc)
        result.attempts.append(attempt)
    best = select_best(result.attempts)
    summary["successful_starts"] = sum(a["success"] for a in result.attempts)
    if best is None:
        summary.update(status="fit_failed", message="No successful finite optimization")
        return result
    mass = 10**best["log_m200"]
    concentration = best["concentration"]
    mass_hit, concentration_hit = bound_hits([best["log_m200"], concentration], config)
    summary.update(status="success", message=best["message"], log_m200=best["log_m200"], m200=mass,
                   concentration=concentration, chi2=best["chi2"], reduced_chi2=best["chi2"]/summary["dof"],
                   mass_bound_hit=mass_hit, concentration_bound_hit=concentration_hit,
                   agreeing_starts=sum(a["success"] and a["chi2"]-best["chi2"] < config.agreement_delta_chi2 for a in result.attempts))
    halo = HALOS[model].velocity(radius, mass, concentration, config.g, config.h0)
    predicted = total_velocity(bar, halo)
    res = residuals(observed, predicted)
    result.predictions = pd.DataFrame(dict(galaxy=name, model=model, radius=radius, v_obs=observed,
        v_err=error, v_bar_squared=bar, v_halo=halo, v_model=predicted, residual=res, normalized_residual=res/error))
    return result

def fit_sample(joined_data, models=("nfw", "burkert"), config=None):
    """Fit each galaxy/model sequentially, retaining invalid and failed cases."""
    config = config or FitConfig()
    return [fit_galaxy(galaxy, model, config)
            for _, galaxy in joined_data.groupby("galaxy", sort=True, dropna=False) for model in models]
