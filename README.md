# SPARC halo fits

Fit NFW and Burkert halos to SPARC rotation curves and reproduce the report figures.

## Run the report

1. From the repository root, install the dependencies in your Python environment:

   ```sh
   python -m pip install -r requirements.txt
   ```

2. Open `notebooks/7_report_figures.ipynb` in a notebook editor, select that environment as the kernel, and run all cells. It uses the processed CSVs and baseline fit configuration in `data/processed/`, refits both halos from one centered start, and writes figures and tables to `figures/report/`.
3. For posterior bands and the separation-filtered figures, generate the MCMC chains first. The repository includes summaries and plots. The chain blocks are stored locally. From the repository root, run:

   ```sh
   python -m utils.mcmc_analysis --workers 4
   ```

   Then rerun notebook 7. Galaxies without usable chains are skipped in the posterior calculations.

## Rebuild the inputs

Run `1_data_processing.ipynb` then `4_sample_fitting.ipynb` in `notebooks/` to regenerate the processed data and baseline fits. Also use this sequence if the saved input checksum differs on your checkout. Notebook 6 displays MCMC results and can regenerate chains with `RUN_MCMC = True`. Notebooks 2, 3, and 5 are optional explorations of rotation curves, individual fits, and stellar-mass trends.
