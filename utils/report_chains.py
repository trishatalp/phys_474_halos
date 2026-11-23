"""Read the saved posterior ensembles used in the report figures.

Check each ensemble against the report inputs and stream its blocks in step order.
"""

import json

import numpy as np


def checked_chain(directory, summary, galaxy, model, input_hash, fit_settings):
    """Locate the selected ensemble and check that its data, priors, and fit settings match."""
    key = (galaxy, model)
    if summary is None or key not in summary.index:
        raise ValueError("missing chain summary")
    row = summary.loc[key]
    if row.converged != True:
        raise ValueError("unresolved chain")

    folder = directory / row.chain_path
    settings = json.loads((folder.parent / "settings.json").read_text())
    expected = {
        "galaxy": galaxy,
        "model": model,
        "input_sha256": input_hash,
        "fit": fit_settings,
        "prior": "uniform in log10(M200/Msun) and linear concentration",
    }
    if any(settings[key] != value for key, value in expected.items()):
        raise ValueError("chain provenance mismatch")
    blocks = sorted(folder.glob("block-*.npz"))
    if not blocks:
        raise ValueError("missing chain blocks")
    return row, settings["sampler"]["walkers"], blocks


def chain_blocks(paths, walkers, expected_steps):
    """Yield each block with its starting step and verify the final step count."""
    start = 0
    for path in paths:
        with np.load(path) as saved:
            chain = saved["chain"]
        end = start + len(chain)
        # The checkpoint filename records the cumulative step count.
        saved_end = int(path.stem.split("-")[1])
        if end != saved_end or chain.shape[1:] != (walkers, 2):
            raise ValueError("incomplete chain")
        yield start, chain
        start = end
    if start != expected_steps:
        raise ValueError("incomplete chain")
