"""Run posterior sampling for galaxies with successful baseline fits.

Check the two pilot galaxies against quadrature before writing full-sample results.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from zlib import crc32

import emcee
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_distribution

from models.fitting import HALOS, bound_hits
from models.mcmc import HaloPosterior, SamplerConfig, grid_marginals, sample_posterior
from models.results import FitConfig


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
BASELINE = PROCESSED / "halo_fits"
PILOT = ("DDO154", "CamB")
PROBABILITIES = np.array([0.16, 0.50, 0.84])


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


@contextmanager
def lock_output(output):
    """Allow one active analysis per output directory, including on Windows."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another analysis is using {output}") from exc
        yield


def check_pilot(posterior, chain, discard):
    coarse_axes, coarse_cdf = grid_marginals(posterior, 512)
    quantiles = np.quantile(chain[discard:].reshape(-1, 2), PROBABILITIES, axis=0).T
    for resolution in (1024, 2048, 4096):
        axes, cdf = grid_marginals(posterior, resolution)
        grid_error, chain_error = [], []
        for index, (axis, cumulative, sample_quantiles) in enumerate(zip(axes, cdf, quantiles)):
            grid_quantiles = np.interp(PROBABILITIES, cumulative, axis)
            grid_error.extend(abs(np.interp(grid_quantiles, coarse_axes[index], coarse_cdf[index])
                                  - PROBABILITIES))
            chain_error.extend(abs(np.interp(sample_quantiles, axis, cumulative) - PROBABILITIES))
        if max(grid_error) <= 0.01:
            break
        coarse_axes, coarse_cdf = axes, cdf
    return float(max(grid_error)), float(max(chain_error))


def posterior_intervals(chain, discard, config):
    """Compute marginal quantiles and the probability near each prior boundary."""
    samples = chain[discard:].reshape(-1, 2)
    log_mass, concentration = samples.T
    quantiles = np.quantile(np.column_stack([log_mass, 10**log_mass, concentration]), PROBABILITIES, axis=0)
    # Use equal fractions of the log span to inspect both ends of c's six-decade support.
    coordinates = np.column_stack([log_mass, np.log10(concentration)])
    lower, upper = np.array(config.bounds, dtype=float).T.copy()
    lower[1], upper[1] = np.log10([lower[1], upper[1]])
    width = 0.01 * (upper - lower)
    near_low = np.mean(coordinates <= lower + width, axis=0)
    near_high = np.mean(coordinates >= upper - width, axis=0)
    return quantiles, near_low, near_high


def analyze_fit(task):
    """Check a saved minimum, sample its posterior, and assemble one results row."""
    (galaxy, fit, config, settings, output, metadata, pilot) = task
    name, model = fit["galaxy"], fit["model"]
    original = np.array([fit["log_m200"], fit["concentration"]])
    posterior = HaloPosterior(galaxy, model, config)
    if fit["n_points"] != len(galaxy) or not np.isclose(
        posterior.chi2(original)[0], fit["chi2"], rtol=1e-9, atol=1e-8
    ):
        raise ValueError(f"Baseline fit cannot be reproduced for {name}/{model}")

    seed = crc32(f"{settings.seed}/{name}/{model}".encode())
    folder = output / "chains" / name / model
    folder.mkdir(parents=True, exist_ok=True)
    settings_file = folder / "settings.json"
    metadata = dict(metadata, galaxy=name, model=model, seed=seed, baseline_endpoint=original.tolist(),
                    dispersed_seed=seed ^ 0xffffffff)
    serialized = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if settings_file.exists() and settings_file.read_text() != serialized:
        raise ValueError(f"Settings changed for {name}/{model}. Choose a different output directory.")
    temporary = settings_file.with_suffix(".tmp")
    temporary.write_text(serialized)
    temporary.replace(settings_file)
    chain, log_prob, acceptance, discard, tau, converged = sample_posterior(
        posterior, original, settings, seed, folder / "local")
    grid_error, chain_error = np.nan, np.nan
    ensemble = "local"
    best_sample = chain[np.unravel_index(np.argmax(log_prob), log_prob.shape)]
    if pilot:
        grid_error, chain_error = check_pilot(posterior, chain, discard)
        if not converged or chain_error > 0.035:
            ensemble = "dispersed"
            chain, log_prob, acceptance, discard, tau, converged = sample_posterior(
                posterior, original, settings, metadata["dispersed_seed"],
                folder / "dispersed", dispersed=True)
            _, chain_error = check_pilot(posterior, chain, discard)
            if -2 * np.max(log_prob) < posterior.chi2(best_sample)[0]:
                best_sample = chain[np.unravel_index(np.argmax(log_prob), log_prob.shape)]
        if not converged or grid_error > 0.01 or chain_error > 0.035:
            raise RuntimeError(f"Pilot validation failed for {name}/{model}: "
                               f"converged={converged}, grid error={grid_error:.4g}, "
                               f"chain error={chain_error:.4g}")

    # If a walker finds a lower objective, start a new optimizer run from that draw.
    best, minimum = original, fit["chi2"]
    improved = posterior.chi2(best_sample)[0] < minimum - config.agreement_delta_chi2
    updated = False
    if improved:
        refined = minimize(lambda p: posterior.chi2(p)[0], best_sample,
                           method=config.method, bounds=config.bounds, options=config.optimizer_options)
        if refined.success and np.isfinite(refined.fun) and refined.fun <= posterior.chi2(best_sample)[0] + 1e-8:
            best, minimum = refined.x, float(refined.fun)
            updated = True

    quantiles, near_low, near_high = posterior_intervals(chain, discard, config)
    mass_hit, concentration_hit = bound_hits(best, config)
    report = dict(fit, scipy_refined=updated, minimum_verified=not improved or updated,
                  scipy_log_m200=best[0], scipy_m200=10**best[0], scipy_concentration=best[1],
                  scipy_mass_bound_hit=mass_hit, scipy_concentration_bound_hit=concentration_hit,
                  scipy_chi2=minimum, scipy_reduced_chi2=minimum / fit["dof"],
                  chi2_reference_p=chi2_distribution.sf(minimum, fit["dof"]),
                  bic=minimum + 2 * np.log(len(galaxy)),
                  ensemble=ensemble, converged=converged, steps=len(chain), burn_in=discard,
                  chain_path=str((folder / ensemble).relative_to(output)),
                  acceptance_mean=acceptance.mean(), acceptance_min=acceptance.min(),
                  acceptance_max=acceptance.max(), acceptance_per_walker=json.dumps(acceptance.tolist()),
                  tau_log_m200=tau[0], tau_concentration=tau[1],
                  pilot_grid_cdf_error=grid_error, pilot_chain_cdf_error=chain_error,
                  boundary_sensitive=bool(np.max([near_low, near_high]) >= 0.05))
    for index, param in enumerate(("log_m200", "concentration")):
        report[f"{param}_lower_edge_fraction"] = near_low[index]
        report[f"{param}_upper_edge_fraction"] = near_high[index]
    for label, values in zip(("p16", "p50", "p84"), quantiles):
        for param, value in zip(("log_m200", "m200", "concentration"), values):
            report[f"posterior_{param}_{label}"] = value
    print(f"{name}/{model}  {len(chain):,} steps  {'converged' if converged else 'unresolved'}", flush=True)
    return report


def run_analysis(output, *, pilot_only=False, workers=1):
    """Validate the active baseline, run the pilots, and optionally run the complete sample."""
    output = Path(output).resolve()
    with lock_output(output):
        return _run_analysis(output, pilot_only=pilot_only, workers=workers)


def _run_analysis(output, *, pilot_only, workers):
    from utils.mcmc_figures import export_comparisons, export_pilot_figures

    input_path = PROCESSED / "selected_galaxy_rotation_curves.csv"
    summary_path = BASELINE / "fit_summary.csv"
    metadata = json.loads((BASELINE / "fit_config.json").read_text())
    checksum = digest(input_path)
    if checksum != metadata["input_sha256"]:
        raise ValueError("Rotation curves changed since SciPy fitting. Regenerate the baseline.")
    config = FitConfig(**metadata["fit"])
    settings = SamplerConfig()
    provenance = dict(input_sha256=checksum, fit_summary_sha256=digest(summary_path),
                      fit=config.to_dict(), sampler=asdict(settings), emcee=emcee.__version__,
                      prior="uniform in log10(M200/Msun) and linear concentration", checkpoint_version=1)
    groups = dict(tuple(pd.read_csv(input_path).groupby("galaxy")))
    summary = pd.read_csv(summary_path)
    expected = {(name, model) for name in groups for model in HALOS}
    if len(summary) != len(expected) or set(zip(summary.galaxy, summary.model)) != expected:
        raise ValueError("Baseline must contain exactly one row per galaxy and halo model")
    pilot_tasks, sample_tasks = [], []
    for row in summary.to_dict("records"):
        pilot = row["galaxy"] in PILOT
        if row["status"] == "success":
            task = (groups[row["galaxy"]], row, config, settings, output, provenance, pilot)
            (pilot_tasks if pilot else sample_tasks).append(task)
    if len(pilot_tasks) != len(PILOT) * len(HALOS):
        raise ValueError("Both pilot galaxies require successful baseline fits in both models")
    pilot_tasks.sort(key=lambda task: PILOT.index(task[1]["galaxy"]))
    pilot_results = pd.DataFrame(map(analyze_fit, pilot_tasks))
    pilot_results.to_csv(output / "pilot_summary.csv", index=False)
    export_pilot_figures(pilot_results, groups, config, output)
    if pilot_only:
        return pilot_results
    print("Pilot validation passed. Running the full sample.", flush=True)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            sample_results = list(executor.map(analyze_fit, sample_tasks))
    else:
        sample_results = list(map(analyze_fit, sample_tasks))
    results = pd.concat([pilot_results, pd.DataFrame(sample_results),
                         summary.loc[summary.status != "success"]], ignore_index=True)
    return export_comparisons(results, PROCESSED / "selected_galaxy_properties.csv", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASELINE / "mcmc")
    parser.add_argument("--pilot", action="store_true", help="Validate DDO154 and CamB only")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent galaxy fits")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run_analysis(args.output, pilot_only=args.pilot, workers=args.workers)


if __name__ == "__main__":
    main()
