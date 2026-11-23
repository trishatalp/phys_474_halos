"""Sample the joint posterior of log halo mass and concentration.

Save resumable walker chains and check their marginal distributions against numerical quadrature.
"""

from dataclasses import dataclass

import emcee
import numpy as np
from scipy.special import logsumexp

from .baryons import velocity_squared
from .fitting import chi_squared_batch


@dataclass(frozen=True)
class SamplerConfig:
    walkers: int = 32
    first_steps: int = 4000
    block_steps: int = 2000
    max_steps: int = 50000
    seed: int = 474

    def __post_init__(self):
        if self.walkers < 4 or min(self.first_steps, self.block_steps, self.max_steps) <= 0:
            raise ValueError("Expected at least four walkers and positive step counts")


class HaloPosterior:
    def __init__(self, galaxy, model, config):
        columns = ["radius", "v_obs", "v_err", "v_gas", "v_disk", "v_bulge"]
        radius, observed, error, gas, disk, bulge = (
            galaxy.sort_values("radius")[columns].to_numpy(dtype=float).T
        )
        if (len(radius) <= 2 or np.any(radius <= 0) or np.any(error <= 0)
                or not np.isfinite([radius, observed, error, gas, disk, bulge]).all()):
            raise ValueError("Expected at least three finite measurements with positive radii and errors")
        baryons = velocity_squared(gas, disk, bulge, config.y_disk, config.y_bulge)
        self.args = radius, observed, error, baryons, model, config
        self.lower, self.upper = np.asarray(config.bounds).T

    def chi2(self, points):
        return chi_squared_batch(np.atleast_2d(points), *self.args)

    def __call__(self, points):
        """Unnormalized log posterior, uniform in log mass and linear concentration."""
        points = np.atleast_2d(points)
        inside = np.all((points >= self.lower) & (points <= self.upper), axis=1)
        log_prob = np.full(len(points), -np.inf)
        if np.any(inside):
            chi2 = self.chi2(points[inside])
            log_prob[inside] = np.where(np.isfinite(chi2), -0.5 * chi2, -np.inf)
        return log_prob


def initialize_walkers(posterior, center, count, rng, dispersed=False):
    lower, upper = posterior.lower, posterior.upper
    # A fractional perturbation in c keeps a start near the concentration floor local.
    scale = np.array([0.03, max(center[1] * 0.03, 0.001)])
    points = np.empty((0, 2))
    for _ in range(100):
        if dispersed:
            proposed = rng.uniform(lower, upper, size=(count, 2))
            proposed[:, 1] = np.exp(rng.uniform(np.log(lower[1]), np.log(upper[1]), count))
        else:
            proposed = rng.normal(center, scale, size=(count, 2))
        inside = np.all((proposed > lower) & (proposed < upper), axis=1)
        proposed = proposed[inside]
        points = np.concatenate([points, proposed[np.isfinite(posterior(proposed))]])
        if len(points) >= count:
            return points[:count]
    raise ValueError("Could not initialize finite, interior walkers")


def chain_diagnostics(chain, previous_tau):
    """Estimate burn-in and check the length and stability of the remaining chain."""
    tau = emcee.autocorr.integrated_time(chain, tol=0)
    if not np.isfinite(tau).all() or np.any(tau <= 0):
        return 0, np.full(2, np.nan), False
    discard = int(np.ceil(2 * tau.max()))
    if len(chain) - discard < 50:
        return 0, np.full(2, np.nan), False
    tau = emcee.autocorr.integrated_time(chain[discard:], tol=0)
    stable = np.isfinite(tau).all() and np.all(tau > 0)
    stable = stable and np.all(np.abs(previous_tau / tau - 1) < 0.01)
    return discard, tau, bool(stable and len(chain) - discard > 100 * tau.max())


def sample_posterior(posterior, center, settings, seed, directory, *, dispersed=False):
    """Store atomic blocks and restore the exact ensemble and random state."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    sampler = emcee.EnsembleSampler(settings.walkers, 2, posterior, vectorize=True)
    chains, log_probs = [], []
    accepted = np.zeros(settings.walkers)
    tau = np.full(2, np.nan)
    discard, completed, converged = 0, 0, False
    state = None
    for path in sorted(directory.glob("block-*.npz")):
        with np.load(path) as block:
            chains.append(block["chain"])
            log_probs.append(block["log_prob"])
            accepted += block["accepted"]
            tau, discard, converged = block["tau"], int(block["discard"]), bool(block["converged"])
            completed += len(chains[-1])
            if completed != int(path.stem.split("-")[1]):
                raise ValueError(f"Missing or inconsistent checkpoint before {path}")
            random_state = ("MT19937", block["keys"], int(block["position"]),
                            int(block["has_gauss"]), float(block["gaussian"]))
            state = emcee.State(chains[-1][-1], log_probs[-1][-1], random_state=random_state)
    if state is None:
        points = initialize_walkers(posterior, center, settings.walkers, rng, dispersed)
        state = emcee.State(points, random_state=rng.get_state())

    while completed < settings.max_steps and not converged:
        target = settings.first_steps if not completed else completed + settings.block_steps
        steps = min(target, settings.max_steps) - completed
        sampler.reset()
        state = sampler.run_mcmc(state, steps)
        chains.append(sampler.get_chain())
        log_probs.append(sampler.get_log_prob())
        accepted += sampler.backend.accepted
        completed += steps
        discard, tau, converged = chain_diagnostics(np.concatenate(chains), tau)
        _, keys, position, has_gauss, gaussian = state.random_state
        path = directory / f"block-{completed:06}.npz"
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez(stream, chain=chains[-1], log_prob=log_probs[-1],
                     accepted=sampler.backend.accepted, keys=keys, position=position,
                     has_gauss=has_gauss, gaussian=gaussian, discard=discard, tau=tau,
                     converged=converged)
        temp.replace(path)
    return np.concatenate(chains), np.concatenate(log_probs), accepted / completed, discard, tau, converged


def grid_marginals(posterior, resolution):
    """Return cell coordinates and marginal CDFs using linear-c cell widths."""
    t = np.linspace(0, 1, resolution + 1)
    x = posterior.lower[0] + (posterior.upper[0] - posterior.lower[0]) * (1 - np.cos(np.pi * t)) / 2
    c = np.unique(np.concatenate([
        np.linspace(posterior.lower[1], posterior.upper[1], resolution + 1),
        np.geomspace(posterior.lower[1], posterior.upper[1], resolution + 1),
    ]))
    x_mid, c_mid = (x[:-1] + x[1:]) / 2, (c[:-1] + c[1:]) / 2
    log_density = np.empty((len(x_mid), len(c_mid)))
    for index, mass in enumerate(x_mid):
        log_density[index] = posterior(np.column_stack([np.full_like(c_mid, mass), c_mid]))
    # Grid cells have unequal areas. Integrate in log mass and linear concentration.
    log_weight = log_density + np.log(np.diff(x))[:, None] + np.log(np.diff(c))
    weight = np.exp(log_weight - logsumexp(log_weight))
    mass_cdf = np.r_[0, np.cumsum(weight.sum(axis=1))]
    concentration_cdf = np.r_[0, np.cumsum(weight.sum(axis=0))]
    return (x, c), (mass_cdf, concentration_cdf)
