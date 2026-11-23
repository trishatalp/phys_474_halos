"""Plot posterior diagnostics and compare model preferences across stellar masses.

Read the saved chains for velocity bands and use paired fit summaries for the population figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.fitting import HALOS, total_velocity
from models.mcmc import HaloPosterior
from utils.plotting import plot_model_comparison


LABELS = (r"$\log_{10}(M_{200}/M_\odot)$", "Concentration")
PREFERENCES = ("NFW preferred", "Inconclusive", "Burkert preferred")
COLORS = ("#bc591f", "#898c93", "#26758a")


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{extension}", dpi=200)
    finally:
        plt.close(fig)


def export_pilot_figures(results, galaxies, config, output):
    directory = output / "figures"
    for name, rows in results.groupby("galaxy"):
        predictions = []
        for row in rows.itertuples():
            model = row.model
            best = (row.scipy_log_m200, row.scipy_concentration)
            parts = []
            for path in sorted((output / row.chain_path).glob("block-*.npz")):
                with np.load(path) as block:
                    parts.append(block["chain"])
            chain = np.concatenate(parts)
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5), layout="constrained")
            fig.suptitle(f"{name} · {model.upper()} walker traces")
            # Downsample the plotted lines while retaining the full chain on disk.
            stride = max(1, len(chain) // 2000)
            for index, ax in enumerate(axes):
                ax.plot(np.arange(0, len(chain), stride), chain[::stride, :, index],
                        color="#26758a", alpha=0.2, linewidth=0.5)
                ax.axvline(row.burn_in, color="#bc591f", linestyle="--", label="Burn-in")
                ax.set_ylabel(LABELS[index])
            axes[0].legend(frameon=False)
            axes[-1].set_xlabel("Walker step")
            save_figure(fig, directory, f"{name}_{model}_traces")

            samples = chain[row.burn_in:].reshape(-1, 2)
            fig, axes = plt.subplots(2, 2, figsize=(7, 6), layout="constrained")
            fig.suptitle(f"{name} · {model.upper()} posterior")
            axes[0, 1].set_visible(False)
            for index in range(2):
                ax = axes[index, index]
                ax.hist(samples[:, index], bins=60, density=True, color="#26758a", alpha=0.75)
                for quantile in np.quantile(samples[:, index], [0.16, 0.5, 0.84]):
                    ax.axvline(quantile, color="black", linestyle=":")
                ax.axvline(best[index], color="#bc591f", linestyle="--", label="SciPy minimum")
                ax.set_xlabel(LABELS[index])
                ax.set_ylabel("Density")
            axes[0, 0].legend(frameon=False)
            joint = axes[1, 0]
            joint.hist2d(*samples.T, bins=65, cmap="Blues", cmin=1)
            joint.plot(*best, marker="+", color="#bc591f", markersize=11)
            joint.set(xlabel=LABELS[0], ylabel=LABELS[1])
            save_figure(fig, directory, f"{name}_{model}_corner")

            radius, observed, error, bar, *_ = HaloPosterior(galaxies[name], model, config).args
            velocity = HALOS[model].velocity(radius, 10**best[0], best[1], config.g, config.h0)
            predicted = total_velocity(bar, velocity)
            predictions.append(pd.DataFrame(dict(galaxy=name, model=model, radius=radius,
                                                 v_obs=observed, v_err=error, v_model=predicted)))
        predictions = pd.concat(predictions)
        fig, _ = plot_model_comparison(predictions, name)
        save_figure(fig, directory, f"{name}_rotation_curve")
        fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
        for model, values in predictions.groupby("model"):
            ax.plot(values.radius, (values.v_obs - values.v_model) / values.v_err,
                    marker="o", markersize=3, label=model.upper())
        ax.axhline(0, color="gray", linewidth=1)
        ax.set(title=name, xlabel="Radius (kpc)", ylabel="Normalized residual")
        ax.legend(frameon=False)
        save_figure(fig, directory, f"{name}_residuals")


def export_comparisons(results, properties_path, output):
    properties = pd.read_csv(properties_path)[["galaxy", "stellar_mass_msun"]]
    data = results.merge(properties, on="galaxy", how="left", validate="many_to_one")
    if not (np.isfinite(data.stellar_mass_msun) & (data.stellar_mass_msun > 0)).all():
        raise ValueError("Each fitted galaxy requires a positive finite stellar mass")
    verified = data.loc[data.minimum_verified.eq(True)]
    pairs = verified.pivot(index="galaxy", columns="model", values="bic").reindex(columns=["nfw", "burkert"])
    counts = verified.pivot(index="galaxy", columns="model", values="n_points").reindex(columns=pairs.columns)
    delta = (pairs.nfw - pairs.burkert).where(counts.nfw == counts.burkert).rename("delta_bic")
    comparison = properties.merge(delta, on="galaxy", how="inner").dropna(subset=["delta_bic"])
    comparison["preference"] = np.select(
        [comparison.delta_bic <= -2, comparison.delta_bic >= 2],
        [PREFERENCES[0], PREFERENCES[2]], default=PREFERENCES[1])
    comparison["log_stellar_mass"] = np.log10(comparison.stellar_mass_msun)
    absolute_fit = verified.pivot(index="galaxy", columns="model", values="chi2_reference_p").reindex(columns=pairs.columns)
    comparison = comparison.merge(absolute_fit.add_prefix("chi2_reference_p_"), on="galaxy")
    comparison["both_poor_absolute_fit"] = comparison[["chi2_reference_p_nfw", "chi2_reference_p_burkert"]].lt(0.05).all(axis=1)
    comparison.to_csv(output / "model_preference.csv", index=False)
    data = data.merge(comparison.drop(columns="stellar_mass_msun"), on="galaxy", how="left")
    data.sort_values(["galaxy", "model"]).to_csv(output / "results.csv", index=False)
    if comparison.empty:
        return data

    directory = output / "figures"
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.axhspan(-2, 2, color="#e6e8ec", label="Inconclusive range")
    for preference, color in zip(PREFERENCES, COLORS):
        rows = comparison.loc[comparison.preference == preference]
        ax.scatter(rows.log_stellar_mass, rows.delta_bic, color=color, s=22, label=preference)
    ax.set(xlabel=r"$\log_{10}(M_\star/M_\odot)$", ylabel=r"$\mathrm{BIC}_{\rm NFW}-\mathrm{BIC}_{\rm Burkert}$")
    ax.set_yscale("symlog", linthresh=2)
    ax.legend(frameon=False)
    save_figure(fig, directory, "stellar_mass_delta_bic")

    bins = np.floor(comparison.log_stellar_mass).astype(int).rename("lower_log_mass")
    counts = pd.crosstab(bins, comparison.preference).reindex(columns=PREFERENCES, fill_value=0)
    counts["total"] = counts.sum(axis=1)
    fractions = counts[list(PREFERENCES)].div(counts.total, axis=0)
    counts.join(fractions.add_suffix(" fraction")).to_csv(output / "stellar_mass_bins.csv")
    ax = fractions.plot.bar(stacked=True, color=COLORS, figsize=(8, 5), rot=0)
    ax.set_xticks(np.arange(len(counts)), [f"{edge}–{edge + 1}\nn={n}" for edge, n in counts.total.items()])
    ax.set(xlabel=r"$\log_{10}(M_\star/M_\odot)$", ylabel="Fraction of paired fits", ylim=(0, 1))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)
    ax.figure.tight_layout()
    save_figure(ax.figure, directory, "stellar_mass_binned_preference")
    return data

def export_velocity_bands(pilot, galaxies, config, output):
    """Pointwise latent total-velocity intervals from paired posterior draws."""
    for name, rows in pilot.groupby("galaxy"):
        fig, ax = plt.subplots(layout="constrained")
        for index, row in enumerate(rows.itertuples()):
            radius, observed, error, bar, *_ = HaloPosterior(galaxies[name], row.model, config).args
            if index == 0:
                ax.errorbar(radius, observed, yerr=error, fmt="ko", ms=4, label="SPARC")
            curves = []
            steps = 0
            for path in sorted((output / row.chain_path).glob("block-*.npz")):
                with np.load(path) as block:
                    chain = block["chain"]
                draws = chain[max(0, int(row.burn_in) - steps):].reshape(-1, 2)
                steps += len(chain)
                if steps != int(path.stem.split("-")[1]):
                    raise ValueError(f"Incomplete chain at {path}")
                for start in range(0, len(draws), 4096):
                    mass, concentration = draws[start:start + 4096].T
                    halo = HALOS[row.model].velocity(radius, 10**mass[:, None],
                                                     concentration[:, None], config.g, config.h0)
                    curves.append(total_velocity(bar, halo))
            if steps != row.steps or not curves:
                raise ValueError(f"Missing chain data for {name}/{row.model}")
            low, median, high = np.quantile(np.concatenate(curves), [.16, .5, .84], axis=0)
            color = COLORS[0 if row.model == "nfw" else 2]
            ax.fill_between(radius, low, high, color=color, alpha=.24)
            ax.plot(radius, median, color=color, label=row.model.upper())
        ax.set(title=f"{name} | Posterior median and 68% velocity band",
               xlabel="Radius (kpc)", ylabel="Total circular speed (km/s)")
        ax.legend(frameon=False)
        save_figure(fig, output / "figures", f"{name}_velocity_bands")


def export_precision_scatter(results, output):
    from matplotlib.colors import Normalize

    data = results.dropna(subset=["delta_bic", "log_stellar_mass"]).copy()
    data["width"] = data.posterior_log_m200_p84 - data.posterior_log_m200_p16
    norm = Normalize(0, data.width.max())
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11, 5), layout="constrained")
    for ax, model in zip(axes, ("nfw", "burkert")):
        rows = data.loc[data.model == model]
        ax.axhspan(-2, 2, color="0.91")
        points = ax.scatter(rows.log_stellar_mass, rows.delta_bic, c=rows.width, norm=norm,
                            cmap="viridis", s=25)
        unresolved = rows.loc[rows.converged.eq(False)]
        if len(unresolved):
            ax.scatter(unresolved.log_stellar_mass, unresolved.delta_bic, marker="x",
                       color="black", s=72, label="Convergence unresolved")
            ax.legend(frameon=False)
        ax.set_yscale("symlog", linthresh=2)
        ax.set(title=model.upper(), xlabel=r"$\log_{10}(M_\star/M_\odot)$")
    axes[0].set_ylabel("BIC (NFW) minus BIC (Burkert)")
    fig.colorbar(points, ax=axes, label="68% halo mass interval width (dex)")
    save_figure(fig, output / "figures", "stellar_mass_posterior_precision")


def export_fraction_intervals(bins, output):
    from scipy.stats import beta

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    centers = bins.lower_log_mass + .5
    for offset, category, color in zip((-.16, 0, .16), PREFERENCES, COLORS):
        counts = bins[category]
        low, high = beta.ppf(np.array([[.16], [.84]]), counts + .5, bins.total - counts + 1)
        x = centers + offset
        ax.vlines(x, low, high, color=color, linewidth=2)
        ax.scatter(x, counts / bins.total, color=color, label=category)
    ax.set_xticks(centers, [f"{edge}–{edge+1}\nn={n}" for edge, n in zip(bins.lower_log_mass, bins.total)])
    ax.set(xlabel=r"$\log_{10}(M_\star/M_\odot)$", ylabel="Fraction of galaxies",
           title="Observed fractions and marginal 68% credible intervals", ylim=(-.035, 1.035))
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, output / "figures", "stellar_mass_fraction_intervals")
