# Transdermal Microneedle Diffusion Models

This repository gathers the analytical derivations and Python implementations developed to study drug transport from a dissolving microneedle patch into skin. The work focuses on explicit finite-difference models (1D and 2D) that complement the closed-form analytical solution of the simplest diffusion problem. Broader project context, including the hardware design study and COMSOL simulations, is documented in `final_report_biomechanics.pdf`, but only the analytical and numerical assets live in this repo.

## Repository Layout
```text
analytic/          Analytical series solution and plots for the 1D diffusion model
numerical/         Python implementations of the finite-difference studies
  Numerical 1/     Uniform 1D model mirroring the analytical case
  Numerical 2/     1D model with a finite drug plug (needle-length-limited initial condition)
  Numerical 3/     Layered 1D model for stratum corneum, epidermis, dermis
  Numerical 4/     2D explicit model and post-processing utilities
figures/           Published plots and CSV exports used in the final report
final_report_biomechanics.pdf  Full project background and discussion
video_1.mp4, video_2.mp4       Diffusion animations from the 2D study
```

## Requirements
- Python 3.9+
- NumPy, Matplotlib (install with `python3 -m pip install numpy matplotlib`)
- Large 2D runs in `Numerical 4` additionally expect SciPy-compatible tooling only if you plan to extend the solver (current scripts use NumPy/Matplotlib only).

All scripts are self-contained; no external datasets are required beyond the repository contents. Some post-processing utilities in `Numerical 4/experiments4.py` load `.npz` snapshots that were not committed due to size; see the notes below on regenerating them.

## Running the studies
### Analytical solution (`analytic/Analytic.py`)
- Solves the 1D diffusion equation with Laplace-series solution (Equation 21 in the report).
- Toggle the `experiment_*` calls at the bottom of the script to export the figures discussed in Section 4.1 and 4.2 of the report (png files saved alongside the script).

### Numerical 1 - uniform 1D model (`numerical/Numerical 1/Numerical1.py`)
- Explicit finite-difference scheme with constant initial concentration along the domain.
- Matches the analytical solution (see `figures/figure_1.png`). Uncomment the desired `experiment_*` call to generate:
  - `experiment_0`-`experiment_4`: parameter sweeps for duration, diffusion coefficient, initial concentration, and needle length.
  - `experiment_5`: release profiles vs. time for several diffusion coefficients (`figures/figure_5.png`).

### Numerical 2 - finite plug 1D model (`numerical/Numerical 2/Numerical2.py`)
- Same scheme but only the needle-length segment starts at `C0`; the remainder of the domain is initially drug-free.
- Provides plots comparing time evolution, diffusion coefficients, initial concentrations, and needle lengths (saved as `numerical2_experiment*.png`).

### Numerical 3 - layered 1D model (`numerical/Numerical 3/`)
- `Numerical3.py` implements spatially varying diffusion coefficients to mimic stratum corneum, epidermis, and dermis.
- `experiments3.py` orchestrates parameter sweeps and writes figures (`numerical3_experiment*.png`) used for the delivered-quantity analyses in Section 4.3.
- Adjust the diffusion tensors, layer thicknesses, and time grid in `experiments3.py` to explore new scenarios. Simulations can take several minutes for the denser grids used in the report.

### Numerical 4 - 2D explicit model (`numerical/Numerical 4/`)
- `Numerical4.py` expands the explicit scheme to 2D and provides utilities to integrate concentration and delivered mass over time.
- `experiments4.py` contains plotting helpers for spatial heatmaps, time-lapse videos, and delivered-quantity curves.
  - Large `.npz` concentration tensors (e.g., `C_C13.3.npz`, `C_D1e-7.npz`) were omitted. Regenerate them by uncommenting the solver blocks at the bottom of `experiments4.py`, which call `compute_C_matrix_2D` and `save_C_matrix` with the configurations described in Section 4.3.
  - The precomputed plots `numerical4_diffusion_result.png`, `numerical4_initial.png`, and `numerical4_quantity_delivered.png` summarize the scenarios featured in the report.
- The diffusion animations (`video_1.mp4`, `video_2.mp4`) were rendered with `plot_diffusion_video` after down-sampling selected runs.

## Results highlights
- **Analytical vs. Numerical agreement**: `figure_1.png` overlays the Laplace-series solution with the finite-difference output, confirming the explicit scheme’s accuracy for the simplest model (Section 4.1).
- **Diffusion coefficient dominates release kinetics**: Higher `D` values sharply reduce residual concentration at fixed positions and accelerate drug depletion (`figure_2.png`, `figure_6.png`).
- **Initial concentration and needle length**: Larger `C0` or `L` increase the total drug available but have milder influence on concentration gradients than `D` within the simple homogeneous model (`figure_3.png`, `figure_4.png`).
- **Layered skin response**: The complex 1D model with stratum corneum/epidermis/dermis shows that delivered mass is most sensitive to the dermis diffusion coefficient, while longer needles boost delivery with diminishing returns once the drug reserves concentrate near the tip (`numerical3_experiment*.png`, summarized in Figure 8 and Figure 9 of the report).
- **2D distribution patterns**: Heatmaps and delivered-quantity curves in `numerical4_*.png` capture lateral spreading limits; varying initial concentration shifts overall mass delivered, while low diffusion coefficients or shorter needles confine the payload near the insertion site.

## Data and media
- `figures/analitic_*` and `figures/C_matrix_*` CSV files contain sampled concentration profiles used for publication-quality plots.
- `video_1.mp4` and `video_2.mp4` illustrate temporal evolution of the 2D simulations (coarse sub-sampling for manageable file size).

## Limitations and next steps
- Current solvers use explicit finite differences constrained by the Von Neumann stability criterion; exploring Crank-Nicolson or other implicit schemes (recommended in Section 6 of the report) would reduce runtime and relax grid constraints.
- Drug delivery is injected as an initial concentration plug; modeling a flux boundary at the microneedle tip remains future work (see Section 5.1 critiques).
- COMSOL and CAD investigations are described in the final report but are not part of this repository.

## Further reading
Consult `final_report_biomechanics.pdf` for the complete background on microneedle design choices, material properties, extended COMSOL simulations, and literature references supporting the parameter selections used throughout these scripts.
