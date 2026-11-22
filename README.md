# SPARC halo fits

Reusable physics lives in `models/`, data loading and plotting in `utils/`, and execution/export in `notebooks/`.

## Run

Install `requirements.txt` into the project environment and install `ipykernel` for notebook execution. Select that environment in your notebook editor. Launch notebooks from the repository root or `notebooks/`.

1. `notebooks/1_data_processing.ipynb` exports the raw MRT data to CSV, selects the sample, and estimates stellar masses.
2. `notebooks/2_rotation_curves.ipynb` exports observed curves with stellar mass annotations.
3. `notebooks/3_single_galaxy_fitting.ipynb` inspects DDO154 or another chosen galaxy.
4. `notebooks/4_sample_fitting.ipynb` fits all galaxies and exports the results.
5. `notebooks/fit_results.ipynb` inspects saved diagnostics and figures without refitting.
6. `notebooks/5_stellar_mass_halo_preference.ipynb` plots the fit preference against processed stellar mass.
7. `notebooks/6_miscellaneous_plots.ipynb` shows the selected sample's stellar mass histogram with labeled extrema.

Fit exports are in `data/processed/halo_fits/`; fitting figures are in `figures/halo_fits/`. Rerunning overwrites matching outputs. After changing upstream sample cuts, update the sample notebook's `SAMPLE_CUTS` metadata and regenerate fits. An input SHA-256 ties saved fits to the processed CSV.

## Physics and interpretation

Units are kpc, km/s, and solar masses. NFW and Burkert expose `velocity` and `enclosed_mass`; Burkert concentration is R200/r0. Both are normalized to M200 at R200, defined at 200 times critical density. Values at radius zero are explicitly zero.

Fits preserve the original fixed disk/bulge mass-to-light ratios (0.5/0.7), nine starting points, bounds, and L-BFGS-B objective. The initialization uses maximum observed velocity, not catalog Vflat; the same heuristic is used for both models and starts are clipped to bounds. Signed gas contributions are preserved. Component figures display signed square roots when squared contributions are negative.

Data processing uses the same ratios to estimate stellar mass from total and bulge luminosities and stores the result in `stellar_mass_msun`. The halo preference notebook plots raw NFW-minus-Burkert chi-squared versus log stellar mass. It exports PNG/PDF figures under `figures/halo_preference/` and the plotted table under `data/processed/halo_fits/`. Physical gas mass estimates remain deferred. Halo enclosed masses are provided by the specified halo profiles.

`fit_galaxy` and `fit_sample` return `GalaxyFitResult` records without file I/O. `result_tables` converts records into summary, attempts, and predictions tables. Invalid galaxies remain in the summary; individual failed starts remain in attempts. A successful optimizer status does not establish physical fit quality. Boundary flags use an absolute tolerance of 1e-5 in fitted parameter coordinates. Uncertainties in fitted parameters and variable mass-to-light ratios are not implemented.

## Data tables

The raw CSVs retain the rows and units of the corresponding MRT sources.

| Directory | CSV | Rows | Contents |
| --- | --- | ---: | --- |
| `data/raw` | `T1SPARC_Lelli2016c.csv` | 175 | Full SPARC galaxy catalog, including distances, inclinations, photometry, H I properties, and quality flags. |
| `data/raw` | `T2MassModels_Lelli2016c.csv` | 3,391 | All radial measurements, including observed velocities and uncertainties, baryonic velocity components, and surface brightness. |
| `data/raw` | `Bulges.csv` | 175 | Non-parametric bulge luminosity by galaxy in units of 10⁹ solar luminosities. |
| `data/processed` | `selected_galaxy_properties.csv` | 108 | One row per selected galaxy with inclination, quality, luminosities, and estimated stellar mass in solar masses. |
| `data/processed` | `selected_galaxy_rotation_curves.csv` | 2,313 | One row per selected radial measurement with its galaxy's inclination, quality, and luminosities. Input to the halo fits. |

Fit exports are in `data/processed/halo_fits/`. The summary, optimizer attempts, radial predictions, initialization diagnostics, and stellar-mass comparison are saved as CSVs. The fit configuration and input checksum are recorded in `fit_config.json`.



