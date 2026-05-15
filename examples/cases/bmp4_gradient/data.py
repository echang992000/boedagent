from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


try:  # pragma: no cover - exercised in example/tests when numpy is available
    import numpy as np
except ImportError:  # pragma: no cover - local environment may not have numpy
    np = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "bmp4_data" / "bmp4_gradient_data.npz"


@dataclass(frozen=True)
class Bmp4CellLineData:
    name: str
    bmp4_conc: Any
    x_obs: Any
    Rs: Any
    receptor_names: tuple[str, ...]
    ligand_names: tuple[str, ...]
    indices: Any
    x_obs_norm: Any | None = None
    Ls: Any | None = None
    Ls_norm: Any | None = None
    Rs_norm: Any | None = None

    @property
    def min_positive_concentration(self) -> float:
        positive = self.bmp4_conc[self.bmp4_conc > 0]
        if positive.size == 0:
            return 1e-6
        return float(positive.min())

    @property
    def max_concentration(self) -> float:
        return float(self.bmp4_conc.max())


@dataclass(frozen=True)
class Bmp4JointData:
    cell_lines: tuple[str, ...]
    bmp4_conc: Any
    x_obs: Any
    x_obs_norm: Any
    bmp4_conc_norm: Any
    q_obs: Any
    Rs_norm: Any
    kd_prior_shift: Any
    receptor_names: tuple[str, ...]
    ligand_names: tuple[str, ...]

    @property
    def min_positive_concentration(self) -> float:
        positive = self.bmp4_conc[self.bmp4_conc > 0]
        if positive.size == 0:
            return 1e-6
        return float(positive.min())

    @property
    def max_concentration(self) -> float:
        return float(self.bmp4_conc.max())


def load_bmp4_gradient_data(
    path: str | Path | None = None,
) -> dict[str, Bmp4CellLineData]:
    if np is None:  # pragma: no cover - depends on optional numpy
        raise RuntimeError(
            "BMP4 example data loading requires `numpy`. Install the literature "
            "or all extras before running this example."
        )

    archive_path = Path(path or DEFAULT_DATA_PATH)
    with np.load(archive_path, allow_pickle=False) as archive:
        receptor_names = tuple(str(item) for item in archive["receptor_names"].tolist())
        ligand_names = tuple(str(item) for item in archive["ligand_names"].tolist())
        cell_lines = tuple(str(item) for item in archive["cell_lines"].tolist())

        loaded: dict[str, Bmp4CellLineData] = {}
        for cell_line in cell_lines:
            loaded[cell_line] = Bmp4CellLineData(
                name=cell_line,
                bmp4_conc=archive[f"{cell_line}_bmp4_conc"].astype("float32"),
                x_obs=archive[f"{cell_line}_x_obs"].astype("float32"),
                Rs=archive[f"{cell_line}_Rs"][0].astype("float32"),
                receptor_names=receptor_names,
                ligand_names=ligand_names,
                indices=archive[f"{cell_line}_indices"].astype("int32"),
                x_obs_norm=archive[f"{cell_line}_x_obs_norm"].astype("float32"),
                Ls=archive[f"{cell_line}_Ls"].astype("float32"),
                Ls_norm=archive[f"{cell_line}_Ls_norm"].astype("float32"),
                Rs_norm=archive[f"{cell_line}_Rs_norm"].astype("float32"),
            )
        return loaded


def make_log_spaced_grid(
    cell_line_data: Bmp4CellLineData | Bmp4JointData,
    *,
    num_points: int = 128,
) -> Any:
    if np is None:  # pragma: no cover - depends on optional numpy
        raise RuntimeError("BMP4 plotting grid construction requires `numpy`.")
    low = cell_line_data.min_positive_concentration
    high = max(cell_line_data.max_concentration, low * 10.0)
    return np.geomspace(low, high, num=num_points, dtype=np.float32)


def build_joint_bmp4_gradient_data(
    data_bundle: dict[str, Bmp4CellLineData],
    *,
    cell_lines: Sequence[str] | None = None,
    knockdown_shift_value: float = -2.0,
) -> Bmp4JointData:
    if np is None:  # pragma: no cover - depends on optional numpy
        raise RuntimeError("BMP4 joint data construction requires `numpy`.")

    selected = tuple(cell_lines or data_bundle.keys())
    if not selected:
        raise ValueError("At least one BMP4 cell line must be selected.")

    reference = data_bundle[selected[0]]
    receptor_names = reference.receptor_names
    ligand_names = reference.ligand_names

    bmp4_conc = np.stack([data_bundle[name].bmp4_conc for name in selected]).astype("float32")
    x_obs = np.stack([data_bundle[name].x_obs for name in selected]).astype("float32")
    x_obs_norm = np.stack([data_bundle[name].x_obs_norm for name in selected]).astype("float32")
    bmp4_conc_norm = np.stack([data_bundle[name].Ls_norm[:, 0] for name in selected]).astype("float32")
    q_obs = np.stack([data_bundle[name].Rs for name in selected]).astype("float32")
    Rs_norm = np.stack([data_bundle[name].Rs_norm[0] for name in selected]).astype("float32")
    kd_prior_shift = np.zeros((len(selected), len(receptor_names)), dtype="float32")

    receptor_index = {name: idx for idx, name in enumerate(receptor_names)}
    for row, cell_line in enumerate(selected):
        lowered = cell_line.lower()
        if "bmpr2" in lowered and "kd" in lowered:
            kd_prior_shift[row, receptor_index["BMPR2"]] = float(knockdown_shift_value)
        if "acvr1" in lowered and "kd" in lowered:
            kd_prior_shift[row, receptor_index["ACVR1"]] = float(knockdown_shift_value)
        if "bmpr1a" in lowered and "kd" in lowered:
            kd_prior_shift[row, receptor_index["BMPR1A"]] = float(knockdown_shift_value)

    return Bmp4JointData(
        cell_lines=selected,
        bmp4_conc=bmp4_conc,
        x_obs=x_obs,
        x_obs_norm=x_obs_norm,
        bmp4_conc_norm=bmp4_conc_norm,
        q_obs=q_obs,
        Rs_norm=Rs_norm,
        kd_prior_shift=kd_prior_shift,
        receptor_names=receptor_names,
        ligand_names=ligand_names,
    )


__all__ = [
    "Bmp4CellLineData",
    "Bmp4JointData",
    "DEFAULT_DATA_PATH",
    "build_joint_bmp4_gradient_data",
    "load_bmp4_gradient_data",
    "make_log_spaced_grid",
]
