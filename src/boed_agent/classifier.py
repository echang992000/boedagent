"""Module 3 — :class:`DataClassifier`.

Detects whether a dataset is homogeneous enough to be fed to a single
BOED backend call, or whether it should be split into per-cluster
sub-problems.

Two modes:

* ``"raw"`` — cluster the data as given.  A warning is always attached.
* ``"simulator_aware"`` — cluster on summary statistics / residuals
  computed from the simulator.  Preferred when a simulator is available.

Clustering uses HDBSCAN if installed, falling back to Gaussian-Mixture
with BIC-selected ``k`` if ``scikit-learn`` is installed, and finally
to a deterministic "all in one cluster" stub so that the module can
run unit tests in minimal environments.  The backend is pluggable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence


Array = Any  # avoid hard dependency on numpy for the public type hint


@dataclass
class ClassifierConfig:
    mode: str = "simulator_aware"  # "raw" or "simulator_aware"
    max_clusters: int = 8
    min_cluster_size: int = 5
    silhouette_threshold: float = 0.25
    homogeneity_rule: str = "single_cluster_or_low_silhouette"
    feature_fn: Optional[Callable[[Any, Any], Any]] = None
    clusterer: str = "auto"  # one of "auto", "hdbscan", "gmm", "trivial"


@dataclass
class ClassifierResult:
    homogeneous: bool
    cluster_labels: list[int]
    score: float
    method: str
    warnings: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "homogeneous": self.homogeneous,
            "cluster_labels": list(self.cluster_labels),
            "score": float(self.score),
            "method": self.method,
            "warnings": list(self.warnings),
            "notes": dict(self.notes),
        }


class DataClassifier:
    """Triage a dataset into homogeneous / heterogeneous."""

    def __init__(self, mode: str = "simulator_aware", *, config: ClassifierConfig | None = None) -> None:
        config = config or ClassifierConfig(mode=mode)
        config.mode = mode
        self.config = config

    def classify(
        self,
        data: Iterable[Any],
        simulator: Any | None = None,
    ) -> ClassifierResult:
        warnings: list[str] = []
        features = self._features(data, simulator, warnings)
        if not features:
            return ClassifierResult(
                homogeneous=True,
                cluster_labels=[],
                score=1.0,
                method="empty",
                warnings=warnings + ["empty dataset"],
            )

        method, labels = self._cluster(features)
        if len(labels) != len(features):
            # Defensive: align label count to feature count.
            labels = labels[: len(features)] + [0] * (len(features) - len(labels))

        score = _silhouette(features, labels)
        homogeneous = _is_homogeneous(labels, score, self.config)
        return ClassifierResult(
            homogeneous=homogeneous,
            cluster_labels=list(labels),
            score=float(score),
            method=method,
            warnings=warnings,
            notes={
                "n_clusters": len({lbl for lbl in labels if lbl >= 0}),
                "n_points": len(features),
            },
        )

    # --- internals --------------------------------------------------

    def _features(
        self,
        data: Iterable[Any],
        simulator: Any | None,
        warnings: list[str],
    ) -> list[list[float]]:
        data = list(data)
        if self.config.mode == "simulator_aware" and simulator is None:
            warnings.append(
                "simulator_aware mode requested but no simulator supplied; falling "
                "back to raw clustering"
            )
        if self.config.mode == "raw":
            warnings.append(
                "raw clustering: homogeneity depends on the model, not the data"
            )
        fn = self.config.feature_fn
        features: list[list[float]] = []
        for item in data:
            if fn is not None:
                value = fn(item, simulator)
            elif self.config.mode == "simulator_aware" and simulator is not None:
                value = _simulator_residual(item, simulator)
            else:
                value = item
            features.append(_coerce_vector(value))
        return features

    def _cluster(self, features: Sequence[Sequence[float]]) -> tuple[str, list[int]]:
        backend = self.config.clusterer
        if backend in ("auto", "hdbscan"):
            labels = _try_hdbscan(features, self.config)
            if labels is not None:
                return "hdbscan", labels
            if backend == "hdbscan":
                return "hdbscan_unavailable", [0] * len(features)
        if backend in ("auto", "gmm"):
            labels = _try_gmm(features, self.config)
            if labels is not None:
                return "gmm", labels
            if backend == "gmm":
                return "gmm_unavailable", [0] * len(features)
        return "trivial", [0] * len(features)


# --- clustering helpers ------------------------------------------------


def _try_hdbscan(features: Sequence[Sequence[float]], config: ClassifierConfig) -> list[int] | None:
    try:
        import hdbscan  # type: ignore
    except ImportError:
        return None
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return None
    arr = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, config.min_cluster_size))
    return [int(label) for label in clusterer.fit_predict(arr)]


def _try_gmm(features: Sequence[Sequence[float]], config: ClassifierConfig) -> list[int] | None:
    try:
        import numpy as np  # type: ignore
        from sklearn.mixture import GaussianMixture  # type: ignore
    except ImportError:
        return None
    arr = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] < 2:
        return [0] * arr.shape[0]
    best_bic = math.inf
    best_labels: list[int] | None = None
    for k in range(1, min(config.max_clusters, arr.shape[0]) + 1):
        try:
            model = GaussianMixture(n_components=k, random_state=0).fit(arr)
        except Exception:  # pragma: no cover - degenerate data
            continue
        bic = float(model.bic(arr))
        if bic < best_bic:
            best_bic = bic
            best_labels = [int(x) for x in model.predict(arr)]
    return best_labels if best_labels is not None else [0] * arr.shape[0]


def _silhouette(features: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    unique = {lbl for lbl in labels if lbl >= 0}
    if len(features) < 2 or len(unique) < 2:
        return 1.0
    try:
        import numpy as np  # type: ignore
        from sklearn.metrics import silhouette_score  # type: ignore

        arr = np.asarray(features, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return float(silhouette_score(arr, labels))
    except ImportError:
        return _fallback_silhouette(features, labels)


def _fallback_silhouette(features: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    # Pure-Python stand-in based on cluster-mean distances — good enough
    # for test fixtures.
    clusters: dict[int, list[list[float]]] = {}
    for value, label in zip(features, labels):
        clusters.setdefault(int(label), []).append(list(value))
    if len(clusters) < 2:
        return 1.0
    centroids = {
        label: [sum(coord) / len(values) for coord in zip(*values)]
        for label, values in clusters.items()
    }
    total = 0.0
    count = 0
    for label, values in clusters.items():
        for value in values:
            own = _distance(value, centroids[label])
            others = [_distance(value, centroids[o]) for o in clusters if o != label]
            nearest = min(others) if others else own
            denom = max(own, nearest, 1e-9)
            total += (nearest - own) / denom
            count += 1
    return total / max(count, 1)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _simulator_residual(point: Any, simulator: Any) -> list[float]:
    fn = getattr(simulator, "summary", None)
    if callable(fn):
        value = fn(point)
    elif hasattr(simulator, "metadata") and callable(simulator):
        # Use simulator forward evaluation at a default design as a residual.
        try:
            value = simulator(point, None)
        except Exception:
            value = point
    else:
        value = point
    return _coerce_vector(value)


def _coerce_vector(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        flat: list[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                flat.append(float(item))
            elif hasattr(item, "__iter__"):
                flat.extend(_coerce_vector(item))
        return flat
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return [0.0]


def _is_homogeneous(
    labels: Sequence[int], score: float, config: ClassifierConfig
) -> bool:
    unique = {lbl for lbl in labels if lbl >= 0}
    if len(unique) <= 1:
        return True
    if config.homogeneity_rule == "single_cluster_or_low_silhouette":
        return score < config.silhouette_threshold
    return False


__all__ = ["ClassifierConfig", "ClassifierResult", "DataClassifier"]
