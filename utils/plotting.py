"""Reusable plotting helpers for SPARC rotation curves."""

import matplotlib.pyplot as plt


def plot_rotation_curve(galaxy):
    """Return a figure and axes for one galaxy's observed rotation curve.

    Accept rows with processed stellar mass attached. Error bars show the
    supplied SPARC velocity uncertainties, excluding inclination systematics.
    The caller controls displaying, saving, and closing the figure.
    """
    if galaxy.empty or galaxy["galaxy"].nunique() != 1:
        raise ValueError("Provide measurements for exactly one galaxy")
    galaxy = galaxy.sort_values("radius")
    properties = galaxy.iloc[0]

    mass = properties["stellar_mass_msun"]
    fig, ax = plt.subplots(figsize=(7.2, 5.1), layout="constrained")
    ax.errorbar(
        galaxy["radius"], galaxy["v_obs"], yerr=galaxy["v_err"],
        fmt="o", markersize=4.5, color="#176B91",
        markeredgecolor="white", markeredgewidth=0.5,
        ecolor="#5595AE", elinewidth=1.2, capsize=2.5,
        label="Observed (SPARC)", zorder=3,
    )
    ax.set_title(str(properties["galaxy"]), loc="left", fontsize=16, weight="bold", pad=43)
    ax.text(
        0, 1.025,
        f"D = {properties['distance']:g} Mpc  |  "
        f"i = {properties['inclination']:g} ± {properties['inclination_error']:g}°  |  "
        f"Quality {int(properties['quality_flag'])}  |  {len(galaxy)} measurements\n"
        + rf"Estimated stellar mass: $M_\star = {mass:.2e}\,M_\odot$"
        + r"  ($\Upsilon_{\rm disk}=0.5$, $\Upsilon_{\rm bulge}=0.7$)",
        transform=ax.transAxes, fontsize=9, color="#555555",
    )
    ax.set_xlabel("Galactocentric radius (kpc)", fontsize=11)
    ax.set_ylabel("Circular velocity (km/s)", fontsize=11)
    ax.set_xlim(left=0)
    # Include zero, but retain any uncertainty bars extending below it.
    lower_error = (galaxy["v_obs"] - galaxy["v_err"]).min()
    if lower_error >= 0:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(bottom=min(ax.get_ylim()[0], 0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best", fontsize=9)
    return fig, ax


def plot_halo_fit(galaxy, fit_result, config, initial_params=None):
    """Component and residual panels. Signed sqrt displays negative contributions."""
    import numpy as np
    from models.baryons import components_squared
    from models.fitting import HALOS, total_velocity
    if fit_result.summary["status"] != "success":
        raise ValueError("A successful fit is required")
    data = galaxy.sort_values("radius")
    predictions = fit_result.predictions.sort_values("radius")
    radius = predictions["radius"]
    fig, (ax, residual_ax) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    ax.errorbar(radius, predictions.v_obs, yerr=predictions.v_err, fmt="o", color="black", markersize=3, label="Observed")
    components = components_squared(data.v_gas, data.v_disk, data.v_bulge, config.y_disk, config.y_bulge)
    components["baryons"] = predictions.v_bar_squared.to_numpy()
    for name, squared in components.items():
        ax.plot(radius, np.sign(squared)*np.sqrt(np.abs(squared)), linestyle=":", label=name.capitalize() + " (signed)")
    ax.plot(radius, predictions.v_halo, "--", label=fit_result.summary["model"].upper() + " halo")
    ax.plot(radius, predictions.v_model, linewidth=2, label=rf"Optimized total ($\chi^2={fit_result.summary['chi2']:.2f}$)")
    if initial_params is not None:
        halo = HALOS[fit_result.summary["model"]].velocity(radius, 10**initial_params[0], initial_params[1], config.g, config.h0)
        try:
            initial = total_velocity(predictions.v_bar_squared, halo)
            initial_chi2 = float(np.sum(((predictions.v_obs - initial)/predictions.v_err)**2))
            ax.plot(radius, initial, "--", color="gray", alpha=0.7,
                    label=rf"Initial total ($\chi^2={initial_chi2:.2f}$)")
        except ValueError:
            ax.text(0.02, 0.03, "Initial total is invalid", transform=ax.transAxes)
    ax.set_title(f"{fit_result.summary['galaxy']} | {fit_result.summary['model'].upper()} | reduced chi² = {fit_result.summary['reduced_chi2']:.2f}")
    ax.set_ylabel("Circular velocity (km/s)")
    ax.legend(fontsize=8, ncol=2)
    residual_ax.axhline(0, color="gray", linewidth=1)
    residual_ax.errorbar(radius, predictions.normalized_residual, yerr=1, fmt="o", markersize=3)
    residual_ax.set_ylabel("Residual / error")
    residual_ax.set_xlabel("Radius (kpc)")
    for axis in (ax, residual_ax):
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    return fig, (ax, residual_ax)


def plot_model_comparison(predictions, galaxy_name):
    """Compare saved total predictions without rerunning fits."""
    data = predictions.loc[predictions.galaxy == galaxy_name]
    if data.empty:
        raise ValueError("No successful predictions for this galaxy")
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    observed = data.drop_duplicates("radius").sort_values("radius")
    ax.errorbar(observed.radius, observed.v_obs, yerr=observed.v_err, fmt="o", color="black", markersize=3, label="Observed")
    for model, rows in data.groupby("model"):
        rows = rows.sort_values("radius")
        ax.plot(rows.radius, rows.v_model, label=model.upper() + " + baryons")
    ax.set(title=galaxy_name, xlabel="Radius (kpc)", ylabel="Circular velocity (km/s)")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig, ax


def plot_fit_summary(summary):
    """Paired fit-quality comparison and convergence counts; no inference claims."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    success = summary.loc[summary.status == "success"]
    paired = success.pivot(index="galaxy", columns="model", values="reduced_chi2")
    if {"nfw", "burkert"}.issubset(paired.columns):
        paired = paired.dropna()
        axes[0].scatter(paired.nfw, paired.burkert, s=18, alpha=0.7)
        axes[0].set_xscale("symlog", linthresh=0.1)
        axes[0].set_yscale("symlog", linthresh=0.1)
        upper = max(1, paired.max().max()) if len(paired) else 1
        axes[0].plot([0, upper], [0, upper], ":", color="gray")
    axes[0].set(xlabel="NFW reduced chi²", ylabel="Burkert reduced chi²", title="Paired successful fits")
    summary.groupby(["model", "status"]).size().unstack(fill_value=0).plot.bar(ax=axes[1], stacked=True)
    axes[1].set(title="Fit status", ylabel="Galaxies", xlabel="Halo model")
    return fig, axes


def plot_halo_preference_vs_stellar_mass(comparison, *, centered=False, encoding="mass"):
    """Plot halo preference for one selection with scales shared across selections."""
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ratio = comparison["innermost_radius_over_disk_scale"]
    colors = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=ratio.max())
    filled = comparison.inner_point_count.ge(3)
    if centered:
        title, delta, prefix = "Centered start only", "centered_delta_chi2", "centered"
    else:
        title, delta, prefix = "Best of all starts", "delta_chi2", "best"
    if encoding == "mass":
        groups = ((comparison[delta].notna(), "o", True, ""),)
        detail = "All galaxies"
    elif encoding == "coverage":
        groups = (
            (filled, "o", True, r"At least 3 points within $2R_{\rm disk}$"),
            (~filled, "o", False, r"Fewer than 3 points within $2R_{\rm disk}$"),
        )
        detail = "Inner coverage"
    else:
        raise ValueError(f"Unknown marker encoding: {encoding}")

    handles = []
    for mask, marker, fill, label in groups:
        rows = comparison.loc[comparison[delta].notna() & mask]
        rgba = colors(norm(ratio.loc[rows.index]))
        ax.scatter(rows.log_stellar_mass, rows[delta], s=33, marker=marker,
                   edgecolors=rgba, facecolors=rgba if fill else "none", linewidths=1)
        handles.append(Line2D([], [], marker=marker, linestyle="none", color="#414141",
                              markerfacecolor="#414141" if fill else "none", label=label))
    ax.axhline(0, color="#555555", linestyle="--", linewidth=1, zorder=0)
    ax.set(title=f"{title} ({detail})", xlabel=r"$\log_{10}(M_\star/M_\odot)$",
           ylabel=r"$\Delta\chi^2 = \chi^2_{\rm NFW} - \chi^2_{\rm Burkert}$")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    plotted = comparison[["delta_chi2", "centered_delta_chi2"]].stack()
    eligible = comparison[["delta_chi2", "centered_delta_chi2"]].notna().any(axis=1)
    if not plotted.empty:
        masses = comparison.loc[eligible, "log_stellar_mass"]
        ax.update_datalim([
            (masses.min(), min(0, plotted.min())),
            (masses.max(), max(0, plotted.max())),
        ])
        ax.autoscale_view()
    fig.suptitle("Halo fit preference by stellar mass\nPositive favors Burkert | Negative favors NFW")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colors), ax=ax,
                 label="Innermost measured radius / disk scale length")
    if encoding == "coverage":
        fig.legend(handles=handles, loc="outside lower center", ncols=2)
    return fig, ax


def plot_initial_optimized_comparison(fit_results, config, initial_params):
    """Legacy-style initial/best total curves with raw chi-squared in legends.

    Pass one result for a single-model comparison or both for a combined plot.
    Curves use the same observations and baryonic contribution as each fit.
    """
    import numpy as np
    from models.fitting import HALOS, total_velocity
    results = [r for r in fit_results if r.summary["status"] == "success"]
    if not results:
        raise ValueError("At least one successful fit is required")
    if len({r.summary["galaxy"] for r in results}) != 1:
        raise ValueError("All fits must refer to the same galaxy")
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    observed = results[0].predictions.sort_values("radius")
    ax.errorbar(observed.radius, observed.v_obs, yerr=observed.v_err,
                fmt="o", color="black", capsize=3, label="Observed (SPARC)", zorder=5)
    for result in results:
        rows = result.predictions.sort_values("radius")
        model = result.summary["model"]
        name = {"nfw": "NFW", "burkert": "Burkert"}[model]
        color = {"nfw": "tab:orange", "burkert": "tab:blue"}[model]
        halo = HALOS[model].velocity(rows.radius, 10**initial_params[0],
                                    initial_params[1], config.g, config.h0)
        try:
            initial = total_velocity(rows.v_bar_squared, halo)
        except ValueError:
            ax.plot([], [], "--", color=color, label=f"Initial {name}: invalid total velocity")
        else:
            chi2 = float(np.sum(((rows.v_obs - initial)/rows.v_err)**2))
            ax.plot(rows.radius, initial, "--", color=color, linewidth=1.5,
                    label=rf"Initial {name} + Baryons ($\chi^2 = {chi2:.2f}$)")
        ax.plot(rows.radius, rows.v_model, "-", color=color, linewidth=2.5,
                label=rf"Optimized {name} + Baryons ($\chi^2 = {result.summary['chi2']:.2f}$)")
    names = " vs ".join({"nfw": "NFW", "burkert": "Burkert"}[r.summary["model"]] for r in results)
    ax.set(title=f"{results[0].summary['galaxy']}: {names} (Initial vs Optimized)",
           xlabel="Radius (kpc)", ylabel="Circular velocity (km/s)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return fig, ax


def plot_initialization_diagnostics(report):
    """Counts of optimizer outcomes, grouped by starting point."""
    models = report.model.unique()
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 4*len(models)), squeeze=False, layout="constrained")
    for ax, model in zip(axes[:, 0], models):
        rows = report.loc[report.model == model].sort_values("start")
        bottom = 0
        for column, label, color in [("interior_no_contact", "Interior, no contact", "#287D8E"),
                                     ("interior_touched_bound", "Interior, touched bound", "#82C9CD"),
                                     ("successful_bound_hits", "Converged, at bound", "#E5A340"),
                                     ("failures", "Failed", "#BD4B4B")]:
            heights = rows[column]
            ax.bar(rows.start, heights, bottom=bottom, label=label, color=color)
            bottom = bottom+heights
        ax.plot(rows.start, rows.best_matches, "ko--", markersize=4, label="Matches best chi²")
        labels = [f"{r.start}: {r.mass_factor:g} × {r.concentration_factor:g}" + ("\nCENTERED" if r.mass_factor == r.concentration_factor == 1 else "") for r in rows.itertuples()]
        ax.set_xticks(rows.start, labels)
        ax.set(title=model.upper(), ylabel="Number of fits", ylim=(0, rows.attempts.max()*1.15), xlabel="Start: mass factor × concentration factor")
        ax.legend(fontsize=8, ncol=5, loc="upper center")
    return fig, axes


def plot_parameter_bound_counts(counts):
    """Count final parameter endpoints at each bound, by optimizer start."""
    models = counts.model.unique()
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.4 * len(models)),
                             sharey=True, squeeze=False, layout="constrained")
    for model, panels in zip(models, axes):
        for parameter, ax in zip(("log_m200", "concentration"), panels):
            rows = counts.loc[(counts.model == model) & (counts.parameter == parameter)]
            for side, offset, color in [("lower", -0.19, "#287D8E"), ("upper", 0.19, "#BD7240")]:
                values = rows.loc[rows.bound == side].sort_values("start")
                bars = ax.bar(values.start + offset, values.hits, width=0.37,
                              label=f"{side.title()} ({values.limit.iloc[0]:g})", color=color)
                ax.bar_label(bars, labels=[str(n) if n else "" for n in values.hits], padding=2)
            name = "log10 halo mass" if parameter == "log_m200" else "Concentration"
            ax.set(title=f"{model.upper()} · {name}", xlabel="Initialization (5 = centered)",
                   ylabel="Successful fits at bound")
            ax.set_xticks(sorted(rows.start.unique()))
            ax.set_ylim(0, max(1, counts.hits.max()) * 1.25)
            ax.legend()
    return fig, axes
