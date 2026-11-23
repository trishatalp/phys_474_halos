"""Compare optimizer starts across galaxies and halo models.

Track contact with parameter bounds and count starts that recover the best chi-squared.
"""

import numpy as np
from models.fitting import bound_hits

def initialization_diagnostics(attempts, summary, config):
    if "evaluated_at_bound" not in attempts:
        raise ValueError("Rerun sample fitting to record bound contacts during optimization")
    detail = attempts.copy()
    if detail.duplicated(["galaxy", "model", "start"]).any():
        raise ValueError("Duplicate optimizer attempts")

    # Starts are numbered from one in the same nested order as initial_guesses.
    factors = [(m, c) for m in config.factors for c in config.factors]
    detail["mass_factor"] = [factors[int(i)-1][0] for i in detail.start]
    detail["concentration_factor"] = [factors[int(i)-1][1] for i in detail.start]
    detail["centered"] = (detail.mass_factor == 1) & (detail.concentration_factor == 1)
    finite_endpoint = np.isfinite(detail[["chi2", "log_m200", "concentration"]]).all(axis=1)
    detail["converged"] = detail.success & finite_endpoint

    # A fit can touch a bound during optimization and still finish in the interior.
    hits = [bound_hits([r.log_m200, r.concentration], config) for r in detail.itertuples()]
    detail["mass_bound_hit"] = [h[0] for h in hits]
    detail["concentration_bound_hit"] = [h[1] for h in hits]
    detail["any_bound_hit"] = detail.mass_bound_hit | detail.concentration_bound_hit
    detail["converged_at_bound"] = detail.converged & detail.any_bound_hit
    detail["converged_interior"] = detail.converged & ~detail.any_bound_hit
    detail["interior_touched_bound"] = detail.converged_interior & detail.evaluated_at_bound
    detail["interior_no_contact"] = detail.converged_interior & ~detail.evaluated_at_bound

    best = summary[["galaxy", "model", "chi2"]].rename(columns={"chi2": "best_chi2"})
    detail = detail.merge(best, on=["galaxy", "model"], validate="many_to_one")
    detail["matches_best"] = detail.converged & ((detail.chi2-detail.best_chi2).abs() < config.agreement_delta_chi2)
    report = detail.groupby(["model", "start", "mass_factor", "concentration_factor"], as_index=False).agg(
        attempts=("start", "size"), successes=("converged", "sum"),
        mass_bound_hits=("mass_bound_hit", "sum"), concentration_bound_hits=("concentration_bound_hit", "sum"),
        any_bound_hits=("any_bound_hit", "sum"), successful_bound_hits=("converged_at_bound", "sum"),
        interior_touched_bound=("interior_touched_bound", "sum"),
        interior_no_contact=("interior_no_contact", "sum"),
        interior_successes=("converged_interior", "sum"), best_matches=("matches_best", "sum"))
    report["failures"] = report.attempts-report.successes
    for count in ["successes", "any_bound_hits", "best_matches"]:
        report[count+"_percent"] = 100*report[count]/report.attempts
    return detail, report
