# BMP4 Gradient Data (`bmp4_gradient_data.npz`)

BMP4 single-ligand dose-response gradient extracted from `combo_array.npy` for all 4 cell lines. Each cell line has **11 experimental conditions** where only BMP4 concentration varies (all other ligands are zero or at baseline).

## Source

Extracted from `combo_array.npy` (940 × 11 array). The 940 rows are split into 4 cell-line blocks of 235 each. The BMP4 gradient occupies **relative indices 180–190** within each block.

| Cell Line   | Global Indices | BMP4 Conc Range         |
|-------------|----------------|--------------------------|
| NMuMG       | 180–190        | 0.0003 – 3200 ng/mL     |
| BMPR2_KD    | 415–425        | 0 – 3000 ng/mL          |
| ACVR1_KD    | 650–660        | 0 – 3000 ng/mL          |
| BMPR1A_KD   | 885–895        | 0.3858 – 3000 ng/mL     |

Concentrations are ordered **high → low** (index 0 is the highest BMP4 concentration).

## NPZ Keys

**Metadata:**
- `cell_lines` — `["NMuMG", "BMPR2_KD", "ACVR1_KD", "BMPR1A_KD"]`
- `ligand_names` — `["BMP4", "BMP7", "BMP9", "BMP10", "GDF5"]`
- `receptor_names` — `["ACVR1", "BMPR1A", "ACVR2A", "ACVR2B", "BMPR2"]`

**Per cell line** (replace `<CL>` with e.g. `NMuMG`, `BMPR2_KD`, etc.):

| Key | Shape | Description |
|-----|-------|-------------|
| `<CL>_indices` | `(11,)` int32 | Global row indices into `combo_array.npy` |
| `<CL>_x_obs` | `(11,)` | Observed flow cytometry response (raw) |
| `<CL>_Ls` | `(11, 5)` | Ligand concentrations (raw) |
| `<CL>_Rs` | `(11, 5)` | Receptor expression levels (raw) |
| `<CL>_bmp4_conc` | `(11,)` | BMP4 concentration only (`Ls[:, 0]`) |
| `<CL>_x_obs_norm` | `(11,)` | Normalized flow cytometry response |
| `<CL>_Ls_norm` | `(11, 5)` | Normalized ligand concentrations |
| `<CL>_Rs_norm` | `(11, 5)` | Normalized receptor expression levels |

## Normalization

Matches `bmp_simformer_mle.py` normalization pipeline:

- **L (ligands):** Gamma CDF fit on `noised_Ls_4k.npy` → inverse-normal transform (`normalize_via_gamma_cdf` → `gamma_cdf_to_gauss`)
- **R (receptors):** Truncated log-normal → Gaussian (`trunc_lognormal_to_gauss`, μ=0.75, σ=1.5, hi=5)
- **x (observations):** Gamma CDF fit on `sim_x_fat_Rs_noised_Ls_4k.npy` (transposed to N×1×940) → inverse-normal transform

## Usage

```python
import numpy as np

d = np.load("data/bmp4_gradient_data.npz")

for cl in d["cell_lines"]:
    conc = d[f"{cl}_bmp4_conc"]
    obs = d[f"{cl}_x_obs"]
    obs_norm = d[f"{cl}_x_obs_norm"]
    # dose-response curve: plt.plot(conc, obs)
```

## Regeneration

```bash
python extract_bmp4_gradients.py
```
