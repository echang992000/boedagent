from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

np = pytest.importorskip("numpy", exc_type=ImportError)
torch = pytest.importorskip("torch", exc_type=ImportError)

from boed_agent.literature.llm_client import RecordingLLMClient
from boed_agent.prior_builder import AugmentedPrior, DistributionSpec
from examples.agent import bmp4_gradient_agent as bmp4_agent
from examples.agent import bmp4_gradient_comparison
from examples.agent.bmp4_gradient_agent import run_bmp4_gradient_example
from examples.cases.bmp4_gradient.data import (
    DEFAULT_DATA_PATH,
    build_joint_bmp4_gradient_data,
    load_bmp4_gradient_data,
)
from examples.cases.bmp4_gradient.promisys_sequential import (
    initialize_promisys_prior_samples,
    run_promisys_sequential_workflow,
    snap_to_nearest_unused_design,
)
from examples.cases.bmp4_gradient.sequential_analysis import (
    analyze_comparison_run,
    analyze_sequential_run,
)
from examples.cases.bmp4_gradient.promisys_hyperparams import PromisysHyperparams
from examples.cases.bmp4_gradient import promisys_onestep
from examples.cases.bmp4_gradient import promisys_twostep
from examples.cases.bmp4_gradient.pyro import multireceptor as multireceptor_model
from examples.cases.bmp4_gradient.pyro import multireceptor_hierarchical as multireceptor_hierarchical_model
from examples.cases.bmp4_gradient.priors import (
    build_hill_prior,
    build_multireceptor_hierarchical_prior,
    build_multireceptor_prior,
)
from scripts.bmp4_autoresearch import score_run_dir


ROOT = Path(__file__).resolve().parents[1]
PROBLEM_PATH = ROOT / "examples" / "cases" / "bmp4_gradient" / "problem.json"

_PRIOR_LIBRARY = {
    "bottom": {"distribution": "Normal", "params": {"loc": 0.0, "scale": 0.5}},
    "top": {"distribution": "Normal", "params": {"loc": 1.0, "scale": 0.5}},
    "ec50": {"distribution": "LogNormal", "params": {"loc": 0.0, "scale": 0.6}},
    "hill_n": {"distribution": "LogNormal", "params": {"loc": 0.0, "scale": 0.4}},
    "sigma": {"distribution": "LogNormal", "params": {"loc": -2.0, "scale": 0.3}},
    "kd": {"distribution": "LogNormal", "params": {"loc": -0.2, "scale": 0.5}},
    "weight": {"distribution": "LogNormal", "params": {"loc": 0.0, "scale": 0.3}},
    "s50": {"distribution": "LogNormal", "params": {"loc": 0.1, "scale": 0.6}},
    "response_hill": {"distribution": "LogNormal", "params": {"loc": 0.0, "scale": 0.4}},
    "sigma_y": {"distribution": "LogNormal", "params": {"loc": -2.0, "scale": 0.3}},
}


def test_load_bmp4_gradient_data_shapes() -> None:
    bundle = load_bmp4_gradient_data(DEFAULT_DATA_PATH)

    assert set(bundle) == {"NMuMG", "BMPR2_KD", "ACVR1_KD", "BMPR1A_KD"}
    nmumg = bundle["NMuMG"]
    assert nmumg.bmp4_conc.shape == (11,)
    assert nmumg.x_obs.shape == (11,)
    assert nmumg.Rs.shape == (5,)
    assert nmumg.indices.shape == (11,)
    assert len(nmumg.receptor_names) == 5
    assert nmumg.receptor_names == (
        "ACVR1",
        "BMPR1A",
        "ACVR2A",
        "ACVR2B",
        "BMPR2",
    )


def test_build_joint_bmp4_gradient_data_shapes() -> None:
    bundle = load_bmp4_gradient_data(DEFAULT_DATA_PATH)
    joint = build_joint_bmp4_gradient_data(bundle, cell_lines=["NMuMG", "BMPR2_KD"])

    assert joint.cell_lines == ("NMuMG", "BMPR2_KD")
    assert joint.bmp4_conc.shape == (2, 11)
    assert joint.x_obs.shape == (2, 11)
    assert joint.q_obs.shape == (2, 5)
    assert joint.x_obs_norm.shape == (2, 11)
    assert joint.bmp4_conc_norm.shape == (2, 11)
    assert joint.Rs_norm.shape == (2, 5)
    assert joint.kd_prior_shift.shape == (2, 5)
    assert joint.kd_prior_shift[0].sum() == pytest.approx(0.0)
    assert joint.kd_prior_shift[1][4] < 0.0


def test_promisys_hyperparams_parse_and_overlay() -> None:
    config = PromisysHyperparams.from_dict(
        {
            "posterior_net": {
                "hidden_dim": 32,
                "layers": 1,
                "activation": "relu",
                "batch_size": 8,
                "learning_rate": 0.002,
                "steps": 10,
                "simulations": 16,
                "posterior_samples": 12,
            },
            "flow": {
                "num_layers": 3,
                "hidden_sizes": [32, 16],
                "num_bins": 6,
                "activation": "silu",
                "use_resnet": False,
                "dropout_rate": 0.1,
                "standardize_theta": True,
            },
            "objective": {
                "fit_steps": 5,
                "flow_learning_rate": 0.003,
                "design_learning_rate": 0.04,
                "eig_outer_samples": 7,
                "eig_inner_samples": 2,
                "infonce_lambda": 0.25,
                "design_dist_init_std": 0.2,
                "design_temperature_scale": 1.5,
                "selector_temperature_final": 0.03,
                "early_stopping_patience": 10,
                "early_stopping_min_delta": 0.01,
            },
            "mcmc": {
                "warmup": 3,
                "samples": 4,
                "proposal_scale": 0.02,
                "prior_std_floor": 0.05,
            },
        }
    )

    flow_config = config.flow_config({"event_shape": (1,), "hidden_sizes": (96, 96)})

    assert config.posterior_net.hidden_dim == 32
    assert config.to_dict()["mcmc"]["samples"] == 4
    assert config.to_dict()["objective"]["design_temperature_scale"] == 1.5
    assert config.to_dict()["objective"]["selector_temperature_final"] == 0.03
    assert config.to_dict()["objective"]["early_stopping_patience"] == 10
    assert config.to_dict()["objective"]["early_stopping_min_delta"] == 0.01
    assert flow_config["event_shape"] == (1,)
    assert flow_config["hidden_sizes"] == (32, 16)
    assert flow_config["standardize_theta"] is True
    with pytest.raises(ValueError, match="Unknown"):
        PromisysHyperparams.from_dict({"posterior_net": {"not_a_knob": 1}})


def test_eig_plot_uses_broken_linear_axis_for_isolated_prior_start() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from examples.cases.bmp4_gradient.plotting import _add_eig_axis, _eig_axis_break_ranges

    eigs = [15.0, 22.4, 22.5, 22.55]
    ranges = _eig_axis_break_ranges(eigs, baseline=22.5)

    assert ranges is not None
    lower_range, upper_range = ranges
    assert lower_range[0] < 15.0 < lower_range[1]
    assert upper_range[0] < 22.4 < upper_range[1]
    assert lower_range[1] < upper_range[0]

    fig = plt.figure()
    grid = fig.add_gridspec(1, 1)
    _add_eig_axis(
        fig,
        grid[0],
        sharex_axis=None,
        steps=[1, 2, 3, 4],
        eigs=eigs,
        baseline=22.5,
        xlabel="Optimization step",
    )

    assert len(fig.axes) == 2
    assert [axis.get_yscale() for axis in fig.axes] == ["linear", "linear"]
    assert fig.axes[1].get_ylim()[1] < fig.axes[0].get_ylim()[0]
    plt.close(fig)


def test_bmp4_autoresearch_scores_synthetic_run(tmp_path: Path) -> None:
    run_leaf = tmp_path / "trial" / "output" / "promisys_twostep" / "prior" / "joint"
    run_leaf.mkdir(parents=True)
    (run_leaf / "eig_optimization_summary.json").write_text(
        json.dumps({"best_eig": 1.25, "best_cell_line": "NMuMG", "best_dose": 0.75}),
        encoding="utf-8",
    )
    (run_leaf / "fit_summary.json").write_text(
        json.dumps(
            {
                "snpe_loss_history": [3.0, 2.0],
                "likelihood_loss_history": [1.0, 0.5],
                "mcmc_sample_count_by_cell": {"NMuMG": 4},
            }
        ),
        encoding="utf-8",
    )
    for filename in (
        "snpe_posterior_samples.pt",
        "mcmc_posterior_samples.pt",
        "posterior_predictive.pt",
        "likelihood_checkpoint.pkl",
    ):
        (run_leaf / filename).write_bytes(b"")

    score = score_run_dir(tmp_path / "trial", timeout_seconds=10.0, runtime_seconds=2.0)

    assert score["status"] == "completed"
    assert score["best_eig"] == 1.25
    assert score["final_snpe_loss"] == 2.0

    (run_leaf / "mcmc_posterior_samples.pt").unlink()
    failed = score_run_dir(tmp_path / "trial")
    assert failed["status"] == "failed"
    assert any("missing artifacts" in reason for reason in failed["reasons"])


def _tiny_promisys_hyperparams() -> dict[str, Any]:
    return {
        "posterior_net": {
            "hidden_dim": 16,
            "layers": 1,
            "activation": "relu",
            "batch_size": 2,
        },
        "mcmc": {
            "proposal_scale": 0.02,
            "prior_std_floor": 0.04,
        },
    }


def test_prior_translation_records_literature_and_qpcr_sources() -> None:
    bundle = load_bmp4_gradient_data(DEFAULT_DATA_PATH)
    nmumg = bundle["NMuMG"]
    bmpr2_kd = bundle["BMPR2_KD"]
    joint = build_joint_bmp4_gradient_data(bundle, cell_lines=["NMuMG", "BMPR2_KD"])

    hill_augmented = AugmentedPrior(
        distributions={
            "bottom": DistributionSpec(
                name="Normal",
                params={"loc": -0.1, "scale": 0.25},
                source="literature",
                reasoning="stub bottom prior",
                cited_papers=["stub-paper"],
            ),
            "ec50": DistributionSpec(
                name="LogNormal",
                params={"loc": 0.3, "scale": 0.4},
                source="literature",
                reasoning="stub ec50 prior",
                cited_papers=["stub-paper"],
            ),
        }
    )
    hill_prior = build_hill_prior(hill_augmented)
    assert hill_prior.sites["bottom"].source == "literature"
    assert hill_prior.sites["bottom"].fallback is False
    assert hill_prior.sites["ec50"].source == "literature"
    assert hill_prior.sites["top"].source == "fallback"
    assert hill_prior.sites["sigma"].fallback is True

    multireceptor_augmented = AugmentedPrior(
        distributions={
            "kd": DistributionSpec(
                name="LogNormal",
                params={"loc": -0.2, "scale": 0.5},
                source="literature",
                reasoning="stub kd prior",
                cited_papers=["stub-paper"],
            ),
            "weight": DistributionSpec(
                name="LogNormal",
                params={"loc": 0.0, "scale": 0.3},
                source="literature",
                reasoning="stub weight prior",
                cited_papers=["stub-paper"],
            ),
        }
    )
    nmumg_prior = build_multireceptor_prior(
        multireceptor_augmented,
        receptor_names=nmumg.receptor_names,
        receptor_qpcr=nmumg.Rs,
    )
    bmpr2_prior = build_multireceptor_prior(
        multireceptor_augmented,
        receptor_names=bmpr2_kd.receptor_names,
        receptor_qpcr=bmpr2_kd.Rs,
    )

    assert nmumg_prior.sites["kd_BMPR1A"].source == "literature"
    assert nmumg_prior.sites["weight_BMPR1A"].source == "literature"
    assert nmumg_prior.sites["top"].source == "fallback"

    abundance_site = nmumg_prior.sites["abundance_BMPR1A"]
    assert abundance_site.source == "cell_line_qpcr"
    assert abundance_site.distribution == "TruncatedLogNormal"
    assert abundance_site.params["low"] == 0.0
    assert abundance_site.params["high"] == 5.0
    assert abundance_site.params["qpcr_value"] == pytest.approx(float(nmumg.Rs[1]))

    nmumg_centers = np.array(
        [nmumg_prior.sites[f"abundance_{name}"].params["center"] for name in nmumg.receptor_names]
    )
    bmpr2_centers = np.array(
        [bmpr2_prior.sites[f"abundance_{name}"].params["center"] for name in bmpr2_kd.receptor_names]
    )
    assert np.any(np.abs(nmumg_centers - bmpr2_centers) > 1e-6)

    hierarchical_prior = build_multireceptor_hierarchical_prior(
        multireceptor_augmented,
        receptor_names=joint.receptor_names,
    )
    assert hierarchical_prior.sites["log_kd"].source == "literature"
    assert hierarchical_prior.sites["log_weight"].source == "literature"
    assert hierarchical_prior.sites["log_kd"].distribution == "Normal"
    assert hierarchical_prior.sites["log_weight"].distribution == "Normal"
    assert len(hierarchical_prior.sites["log_kd"].params["loc"]) == len(joint.receptor_names)
    assert hierarchical_prior.sites["qpcr_intercept"].source == "fallback"
    assert hierarchical_prior.sites["bottom"].distribution == "LogNormal"
    assert hierarchical_prior.sites["top"].distribution == "LogNormal"

    receptor_specific_augmented = AugmentedPrior(
        distributions={
            "kd_BMPR1A": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(100.0), "scale": 0.35},
                source="expert",
                reasoning="expert BMPR1A affinity",
                cited_papers=["expert-table"],
            ),
            "kd_BMPR2": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(0.38), "scale": 0.35},
                source="expert",
                reasoning="expert BMPR2 affinity",
                cited_papers=["expert-table"],
            ),
        }
    )
    receptor_specific_prior = build_multireceptor_hierarchical_prior(
        receptor_specific_augmented,
        receptor_names=joint.receptor_names,
    )
    log_kd_site = receptor_specific_prior.sites["log_kd"]
    assert log_kd_site.source == "mixed_receptor_specific"
    assert log_kd_site.params["loc"][0] == pytest.approx(0.0)
    assert log_kd_site.params["loc"][1] == pytest.approx(math.log(100.0))
    assert log_kd_site.params["loc"][4] == pytest.approx(math.log(0.38))
    assert log_kd_site.params["scale"][1] == pytest.approx(0.35)
    assert log_kd_site.params["scale"][0] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "description_fn",
    [
        multireceptor_model.problem_description,
        multireceptor_hierarchical_model.problem_description,
    ],
)
def test_multireceptor_problem_descriptions_include_spr_to_eqtk_conversion(description_fn) -> None:
    text = description_fn("BMP4 gradient", ("ACVR1", "BMPR1A", "ACVR2A", "ACVR2B", "BMPR2"))
    assert "K_eqtk = 1e-8 / K_d" in text
    assert "K_eqtk = 10 / K_d_nM" in text
    assert "K_eqtk = 10000 / K_d_pM" in text
    assert "Use K_eqtk, not raw K_d" in text


def test_promisys_onestep_parameter_order_and_transform() -> None:
    assert promisys_onestep.COMPLEX_NAMES == (
        "BMP4_ACVR1_ACVR2A",
        "BMP4_ACVR1_ACVR2B",
        "BMP4_ACVR1_BMPR2",
        "BMP4_BMPR1A_ACVR2A",
        "BMP4_BMPR1A_ACVR2B",
        "BMP4_BMPR1A_BMPR2",
    )
    assert promisys_onestep.RAW_PARAMETER_LOW == pytest.approx(1e-4)
    assert promisys_onestep.RAW_PARAMETER_HIGH == pytest.approx(1e2)
    theta_norm = np.linspace(-2.0, 2.0, promisys_onestep.THETA_DIM, dtype=np.float32)
    theta_raw = promisys_onestep.theta_norm_to_raw(theta_norm)
    affinities, efficiencies, sigma_y = promisys_onestep.split_theta_raw(theta_raw)
    assert affinities.shape == (6,)
    assert efficiencies.shape == (6,)
    assert np.isscalar(float(sigma_y))
    assert np.all(theta_raw > 0.0)
    assert np.all(np.diff(theta_raw[: promisys_onestep.BIOPHYSICAL_THETA_DIM]) > 0.0)
    round_trip = promisys_onestep.theta_raw_to_norm(theta_raw)
    assert np.allclose(round_trip, theta_norm, rtol=2e-5, atol=2e-5)
    assert float(theta_raw[-1]) == pytest.approx(
        math.exp(
            promisys_onestep.OBSERVATION_NOISE_LOG_LOC
            + promisys_onestep.OBSERVATION_NOISE_LOG_SCALE * float(theta_norm[-1])
        )
    )

    rng = np.random.default_rng(0)
    sampled_norm = promisys_onestep.sample_theta_norm_prior(rng, (5000, 1))
    sampled_raw = promisys_onestep.theta_norm_to_raw(sampled_norm)
    log_unit = (
        np.log(sampled_raw) - math.log(promisys_onestep.RAW_PARAMETER_LOW)
    ) / (
        math.log(promisys_onestep.RAW_PARAMETER_HIGH)
        - math.log(promisys_onestep.RAW_PARAMETER_LOW)
    )
    assert float(np.mean(log_unit)) == pytest.approx(0.5, abs=0.02)
    assert float(np.std(sampled_norm)) == pytest.approx(1.0, abs=0.05)
    full_sampled_norm = promisys_onestep.sample_theta_norm_prior(rng, (5000, promisys_onestep.THETA_DIM))
    full_sampled_raw = promisys_onestep.theta_norm_to_raw(full_sampled_norm)
    assert float(np.mean(full_sampled_norm[:, -1])) == pytest.approx(0.0, abs=0.05)
    assert float(np.std(full_sampled_norm[:, -1])) == pytest.approx(1.0, abs=0.05)
    assert np.all(full_sampled_raw[:, -1] >= promisys_onestep.OBSERVATION_NOISE_MIN)
    assert np.all(full_sampled_raw[:, -1] <= promisys_onestep.OBSERVATION_NOISE_MAX)


def test_promisys_twostep_parameter_order_and_transform() -> None:
    assert promisys_twostep.DIMER_COMPLEX_NAMES == (
        "BMP4_ACVR1",
        "BMP4_BMPR1A",
    )
    assert promisys_twostep.COMPLEX_NAMES == (
        "BMP4_ACVR1_ACVR2A",
        "BMP4_ACVR1_ACVR2B",
        "BMP4_ACVR1_BMPR2",
        "BMP4_BMPR1A_ACVR2A",
        "BMP4_BMPR1A_ACVR2B",
        "BMP4_BMPR1A_BMPR2",
    )
    assert promisys_twostep.BINDING_PARAMETER_NAMES == (
        "K_BMP4_ACVR1",
        "K_BMP4_BMPR1A",
        "K_BMP4_ACVR1_ACVR2A",
        "K_BMP4_ACVR1_ACVR2B",
        "K_BMP4_ACVR1_BMPR2",
        "K_BMP4_BMPR1A_ACVR2A",
        "K_BMP4_BMPR1A_ACVR2B",
        "K_BMP4_BMPR1A_BMPR2",
    )
    theta_norm = np.linspace(-2.0, 2.0, promisys_twostep.THETA_DIM, dtype=np.float32)
    theta_raw = promisys_twostep.theta_norm_to_raw(theta_norm)
    affinities, efficiencies, sigma_y = promisys_twostep.split_theta_raw(theta_raw)
    assert affinities.shape == (8,)
    assert efficiencies.shape == (6,)
    assert np.isscalar(float(sigma_y))
    assert np.all(theta_raw > 0.0)
    assert np.all(np.diff(theta_raw[: promisys_twostep.BIOPHYSICAL_THETA_DIM]) > 0.0)
    round_trip = promisys_twostep.theta_raw_to_norm(theta_raw)
    assert np.allclose(round_trip, theta_norm, rtol=2e-5, atol=2e-5)
    assert float(theta_raw[-1]) == pytest.approx(
        math.exp(
            promisys_twostep.OBSERVATION_NOISE_LOG_LOC
            + promisys_twostep.OBSERVATION_NOISE_LOG_SCALE * float(theta_norm[-1])
        )
    )


def test_promisys_twostep_expert_prior_maps_to_binding_parameters() -> None:
    prior = AugmentedPrior(
        distributions={
            "kd_BMPR1A": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(80.0), "scale": 0.01},
            ),
            "kd_ACVR2A": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(0.8), "scale": 0.01},
            ),
            "kd_ACVR2B": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(1.6), "scale": 0.01},
            ),
            "kd_BMPR2": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(0.4), "scale": 0.01},
            ),
        }
    )

    config = promisys_twostep._build_twostep_theta_prior(prior)
    assert config["mode"] == "expert_mapped"
    mapped = {
        item["parameter"]: item.get("source_parameter")
        for item in config["parameter_priors"]
        if item["source"] == "literature_prior"
    }
    assert mapped == {
        "K_BMP4_BMPR1A": "kd_BMPR1A",
        "K_BMP4_ACVR1_ACVR2A": "kd_ACVR2A",
        "K_BMP4_ACVR1_ACVR2B": "kd_ACVR2B",
        "K_BMP4_ACVR1_BMPR2": "kd_BMPR2",
        "K_BMP4_BMPR1A_ACVR2A": "kd_ACVR2A",
        "K_BMP4_BMPR1A_ACVR2B": "kd_ACVR2B",
        "K_BMP4_BMPR1A_BMPR2": "kd_BMPR2",
    }

    rng = np.random.default_rng(3)
    samples = np.stack(
        [promisys_twostep.sample_theta_norm_from_prior_config(rng, config) for _ in range(512)],
        axis=0,
    )
    raw = promisys_twostep.theta_norm_to_raw(samples)
    assert float(np.mean(raw[:, 1])) == pytest.approx(80.0, rel=0.03)
    assert float(np.mean(raw[:, 2])) == pytest.approx(0.8, rel=0.03)
    assert float(np.mean(raw[:, 3])) == pytest.approx(1.6, rel=0.03)
    assert float(np.mean(raw[:, 4])) == pytest.approx(0.4, rel=0.03)


def test_promisys_twostep_generic_kd_prior_maps_to_all_binding_parameters() -> None:
    prior = AugmentedPrior(
        distributions={
            "kd": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(2.0), "scale": 0.01},
            ),
        }
    )

    config = promisys_twostep._build_twostep_theta_prior(prior)
    mapped = [
        item
        for item in config["parameter_priors"]
        if item["source"] == "literature_prior"
    ]

    assert config["mode"] == "expert_mapped"
    assert config["mapped_parameter_count"] == 8
    assert {item["source_parameter"] for item in mapped} == {"kd"}
    assert [item["parameter"] for item in mapped] == list(
        promisys_twostep.BINDING_PARAMETER_NAMES
    )


def test_promisys_onestep_expert_prior_maps_to_binding_parameters() -> None:
    prior = AugmentedPrior(
        distributions={
            "kd_ACVR2A": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(0.8), "scale": 0.01},
            ),
            "kd_ACVR2B": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(1.6), "scale": 0.01},
            ),
            "kd_BMPR2": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(0.4), "scale": 0.01},
            ),
        }
    )

    config = promisys_onestep._build_onestep_theta_prior(prior)
    assert config["mode"] == "expert_mapped"
    mapped = {
        item["parameter"]: item.get("source_parameter")
        for item in config["parameter_priors"]
        if item["source"] == "literature_prior"
    }
    assert mapped == {
        "K_BMP4_ACVR1_ACVR2A": "kd_ACVR2A",
        "K_BMP4_ACVR1_ACVR2B": "kd_ACVR2B",
        "K_BMP4_ACVR1_BMPR2": "kd_BMPR2",
        "K_BMP4_BMPR1A_ACVR2A": "kd_ACVR2A",
        "K_BMP4_BMPR1A_ACVR2B": "kd_ACVR2B",
        "K_BMP4_BMPR1A_BMPR2": "kd_BMPR2",
    }

    rng = np.random.default_rng(4)
    samples = np.stack(
        [promisys_onestep.sample_theta_norm_from_prior_config(rng, config) for _ in range(512)],
        axis=0,
    )
    raw = promisys_onestep.theta_norm_to_raw(samples)
    assert float(np.mean(raw[:, 0])) == pytest.approx(0.8, rel=0.03)
    assert float(np.mean(raw[:, 1])) == pytest.approx(1.6, rel=0.03)
    assert float(np.mean(raw[:, 2])) == pytest.approx(0.4, rel=0.03)


def test_promisys_onestep_generic_kd_prior_maps_to_all_binding_parameters() -> None:
    prior = AugmentedPrior(
        distributions={
            "kd": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(2.0), "scale": 0.01},
            ),
        }
    )

    config = promisys_onestep._build_onestep_theta_prior(prior)
    mapped = [
        item
        for item in config["parameter_priors"]
        if item["source"] == "literature_prior"
    ]

    assert config["mode"] == "expert_mapped"
    assert config["mapped_parameter_count"] == 6
    assert {item["source_parameter"] for item in mapped} == {"kd"}
    assert [item["parameter"] for item in mapped] == list(
        promisys_onestep.BINDING_PARAMETER_NAMES
    )


def test_generic_literature_prior_label_keeps_source_family_context() -> None:
    payload = {
        "source_path": (
            "/tmp/artifacts/bmp4_gradient/multireceptor_hierarchical/"
            "joint__NMuMG__BMPR2_KD/literature_prior.json"
        )
    }

    assert (
        bmp4_agent._prior_output_label(payload)
        == "multireceptor_hierarchical_literature_prior"
    )


def test_bmp4_normalizer_round_trips_synthetic_sources(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    source = tmp_path / "normalizer"
    source.mkdir()
    np.save(source / "noised_Ls_4k.npy", rng.gamma(2.0, 3.0, size=(32, 5, 8)).astype("float32"))
    np.save(source / "sim_x_fat_Rs_noised_Ls_4k.npy", rng.gamma(1.5, 0.75, size=(32, 8, 1)).astype("float32"))
    normalizer = promisys_onestep.Bmp4Normalizer.from_source_dir(source, max_fit_samples=256)

    bmp4 = np.array([0.01, 1.0, 10.0], dtype=np.float32)
    bmp4_round_trip = normalizer.denormalize_bmp4(normalizer.normalize_bmp4(bmp4))
    assert np.allclose(bmp4_round_trip, bmp4, rtol=2e-5, atol=2e-5)

    receptors = np.array([0.2, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    receptor_round_trip = normalizer.denormalize_receptors(normalizer.normalize_receptors(receptors))
    assert np.allclose(receptor_round_trip, receptors, rtol=2e-5, atol=2e-5)


def test_design_temperature_targets_raw_dose_std(tmp_path: Path) -> None:
    jax = pytest.importorskip("jax")

    source = _write_synthetic_normalizer_sources(tmp_path)
    normalizer = promisys_twostep.Bmp4Normalizer.from_source_dir(source, max_fit_samples=512)
    joint = build_joint_bmp4_gradient_data(load_bmp4_gradient_data(DEFAULT_DATA_PATH), cell_lines=["NMuMG", "BMPR2_KD"])
    bounds = promisys_twostep._bmp4_design_bounds(joint, normalizer)
    dose_mu = np.asarray(
        [
            0.5 * (bounds["dose_min"] + bounds["dose_max"]),
            0.75 * bounds["dose_min"] + 0.25 * bounds["dose_max"],
        ],
        dtype=np.float32,
    )
    bmp4_norm_mu = normalizer.normalize_bmp4(dose_mu)

    norm_std = promisys_twostep._design_temperature_norm_std(
        normalizer=normalizer,
        bmp4_norm_mu=bmp4_norm_mu,
        target_dose_std=1.0,
        bounds=bounds,
    )
    summary = promisys_twostep._summarize_bmp4_design_distribution(
        normalizer=normalizer,
        bmp4_norm_mu=bmp4_norm_mu,
        bmp4_norm_log_std=np.log(norm_std),
        bounds=bounds,
        sample_key=jax.random.PRNGKey(0),
        sample_count=8192,
    )

    assert np.allclose(summary["dose_std"], np.ones(2), rtol=0.15, atol=0.15)


def test_design_temperature_schedule_starts_broad_and_ends_at_raw_scale(tmp_path: Path) -> None:
    jax = pytest.importorskip("jax")

    source = _write_synthetic_normalizer_sources(tmp_path)
    normalizer = promisys_twostep.Bmp4Normalizer.from_source_dir(source, max_fit_samples=512)
    joint = build_joint_bmp4_gradient_data(load_bmp4_gradient_data(DEFAULT_DATA_PATH), cell_lines=["NMuMG", "BMPR2_KD"])
    bounds = promisys_twostep._bmp4_design_bounds(joint, normalizer)
    bmp4_norm_mu = np.asarray(
        [
            0.5 * (bounds["bmp4_norm_min"] + bounds["bmp4_norm_max"]),
            0.5 * (bounds["bmp4_norm_min"] + bounds["bmp4_norm_max"]),
        ],
        dtype=np.float32,
    )

    initial_norm_std = promisys_twostep._design_temperature_norm_std_schedule(
        normalizer=normalizer,
        bmp4_norm_mu=bmp4_norm_mu,
        step_index=0,
        total_steps=100,
        final_dose_std=1.0,
        bounds=bounds,
    )
    final_norm_std = promisys_twostep._design_temperature_norm_std_schedule(
        normalizer=normalizer,
        bmp4_norm_mu=bmp4_norm_mu,
        step_index=99,
        total_steps=100,
        final_dose_std=1.0,
        bounds=bounds,
    )
    initial_summary = promisys_twostep._summarize_bmp4_design_distribution(
        normalizer=normalizer,
        bmp4_norm_mu=bmp4_norm_mu,
        bmp4_norm_log_std=np.log(initial_norm_std),
        bounds=bounds,
        sample_key=jax.random.PRNGKey(0),
        sample_count=8192,
    )
    final_summary = promisys_twostep._summarize_bmp4_design_distribution(
        normalizer=normalizer,
        bmp4_norm_mu=bmp4_norm_mu,
        bmp4_norm_log_std=np.log(final_norm_std),
        bounds=bounds,
        sample_key=jax.random.PRNGKey(1),
        sample_count=8192,
    )

    assert np.allclose(initial_norm_std, np.ones_like(initial_norm_std), rtol=1e-6, atol=1e-6)
    assert np.all(np.asarray(initial_summary["dose_std"]) > 10.0)
    assert np.all(np.asarray(final_norm_std) < initial_norm_std)
    assert np.allclose(final_summary["dose_std"], np.ones(2), rtol=0.15, atol=0.15)


def test_selector_temperature_anneals_to_clear_choice() -> None:
    initial = promisys_onestep._selector_temperature(
        step_index=0,
        total_steps=5,
        final_temperature=0.02,
    )
    final = promisys_onestep._selector_temperature(
        step_index=4,
        total_steps=5,
        final_temperature=0.02,
    )
    logits = np.asarray([0.0, 0.1], dtype=np.float32)
    initial_probs = promisys_onestep._selector_probs_from_logits(
        logits,
        selector_temperature=initial,
    )
    final_probs = promisys_onestep._selector_probs_from_logits(
        logits,
        selector_temperature=final,
    )

    assert initial == pytest.approx(1.0)
    assert final == pytest.approx(0.02)
    assert initial_probs[1] < 0.55
    assert final_probs[1] > 0.99


def test_promisys_joint_infonce_objective_gradients_are_context_local() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    params = {
        "theta_w": jnp.asarray(0.2, dtype=jnp.float32),
        "dose_w": jnp.asarray(0.4, dtype=jnp.float32),
        "bias": jnp.asarray(-0.1, dtype=jnp.float32),
    }
    batch = {
        "theta": jnp.asarray(
            np.stack(
                [
                    np.linspace(-0.5, 0.5, 4 * promisys_onestep.THETA_DIM).reshape(4, promisys_onestep.THETA_DIM),
                    np.linspace(0.2, 1.2, 4 * promisys_onestep.THETA_DIM).reshape(4, promisys_onestep.THETA_DIM),
                    np.linspace(-1.0, 0.0, 4 * promisys_onestep.THETA_DIM).reshape(4, promisys_onestep.THETA_DIM),
                ],
                axis=0,
            ),
            dtype=jnp.float32,
        ),
        "y": jnp.asarray(
            np.array(
                [
                    [[-0.2], [0.1], [0.2], [0.4]],
                    [[0.3], [0.5], [0.7], [0.9]],
                    [[-0.5], [-0.4], [-0.1], [0.0]],
                ],
                dtype=np.float32,
            )
        ),
        "r_norm": jnp.zeros((3, 5), dtype=jnp.float32),
    }
    design_params = {
        "bmp4_norm_mu": jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32),
        "bmp4_norm_log_std": jnp.log(jnp.asarray([0.25, 0.3, 0.35], dtype=jnp.float32)),
    }
    design_key = jax.random.PRNGKey(7)
    bounds = {"bmp4_norm_min": -2.0, "bmp4_norm_max": 2.0}

    def log_prob_fn(p, y, theta, xi):
        mean = p["bias"] + p["theta_w"] * theta[:, 0] + p["dose_w"] * xi[:, -1]
        return -0.5 * jnp.square(y[:, 0] - mean)

    def utilities_for_params(p):
        return promisys_onestep._joint_multicontext_utilities(
            p,
            design_params,
            batch,
            design_key=design_key,
            log_prob_fn=log_prob_fn,
            bounds=bounds,
            infonce_lambda=0.5,
            inner_samples=2,
        )

    utilities_without_density = promisys_onestep._joint_multicontext_utilities(
        params,
        design_params,
        batch,
        design_key=design_key,
        log_prob_fn=log_prob_fn,
        bounds=bounds,
        infonce_lambda=999.0,
        inner_samples=2,
    )
    assert np.allclose(
        np.asarray(utilities_for_params(params)),
        np.asarray(utilities_without_density),
        rtol=1e-6,
        atol=1e-6,
    )

    objective_grad = jax.grad(lambda p: jnp.sum(utilities_for_params(p)))(params)
    per_cell_grads = [
        jax.grad(lambda p, idx=index: utilities_for_params(p)[idx])(params)
        for index in range(3)
    ]
    summed_per_cell = jax.tree_util.tree_map(lambda *items: sum(items), *per_cell_grads)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b, rtol=1e-5, atol=1e-5), objective_grad, summed_per_cell)
    )

    loss, aux = promisys_onestep._joint_multicontext_infonce_loss(
        params,
        design_params,
        batch,
        design_key=design_key,
        log_prob_fn=log_prob_fn,
        bounds=bounds,
        weights=jnp.ones(3, dtype=jnp.float32) / 3.0,
        infonce_lambda=0.5,
        inner_samples=2,
    )
    design_grads = jax.grad(
        lambda current_design_params: promisys_onestep._joint_multicontext_infonce_loss(
            params,
            current_design_params,
            batch,
            design_key=design_key,
            log_prob_fn=log_prob_fn,
            bounds=bounds,
            weights=jnp.ones(3, dtype=jnp.float32) / 3.0,
            infonce_lambda=0.5,
            inner_samples=2,
        )[0]
    )(design_params)
    assert np.isfinite(float(loss))
    assert np.all(np.isfinite(np.asarray(aux["utilities"])))
    assert np.all(np.isfinite(np.asarray(design_grads["bmp4_norm_mu"])))
    assert np.all(np.isfinite(np.asarray(design_grads["bmp4_norm_log_std"])))


def test_promisys_twostep_joint_infonce_objective_gradients_are_context_local() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    params = {
        "theta_w": jnp.asarray(0.2, dtype=jnp.float32),
        "dose_w": jnp.asarray(0.4, dtype=jnp.float32),
        "bias": jnp.asarray(-0.1, dtype=jnp.float32),
    }
    batch = {
        "theta": jnp.asarray(
            np.stack(
                [
                    np.linspace(-0.5, 0.5, 4 * promisys_twostep.THETA_DIM).reshape(4, promisys_twostep.THETA_DIM),
                    np.linspace(0.2, 1.2, 4 * promisys_twostep.THETA_DIM).reshape(4, promisys_twostep.THETA_DIM),
                    np.linspace(-1.0, 0.0, 4 * promisys_twostep.THETA_DIM).reshape(4, promisys_twostep.THETA_DIM),
                ],
                axis=0,
            ),
            dtype=jnp.float32,
        ),
        "y": jnp.asarray(
            np.array(
                [
                    [[-0.2], [0.1], [0.2], [0.4]],
                    [[0.3], [0.5], [0.7], [0.9]],
                    [[-0.5], [-0.4], [-0.1], [0.0]],
                ],
                dtype=np.float32,
            )
        ),
        "r_norm": jnp.zeros((3, 5), dtype=jnp.float32),
    }
    design_params = {
        "bmp4_norm_mu": jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32),
        "bmp4_norm_log_std": jnp.log(jnp.asarray([0.25, 0.3, 0.35], dtype=jnp.float32)),
    }
    design_key = jax.random.PRNGKey(7)
    bounds = {"bmp4_norm_min": -2.0, "bmp4_norm_max": 2.0}

    def log_prob_fn(p, y, theta, xi):
        mean = p["bias"] + p["theta_w"] * theta[:, 0] + p["dose_w"] * xi[:, -1]
        return -0.5 * jnp.square(y[:, 0] - mean)

    def utilities_for_params(p):
        return promisys_twostep._joint_multicontext_utilities(
            p,
            design_params,
            batch,
            design_key=design_key,
            log_prob_fn=log_prob_fn,
            bounds=bounds,
            infonce_lambda=0.5,
            inner_samples=2,
        )

    utilities_without_density = promisys_twostep._joint_multicontext_utilities(
        params,
        design_params,
        batch,
        design_key=design_key,
        log_prob_fn=log_prob_fn,
        bounds=bounds,
        infonce_lambda=999.0,
        inner_samples=2,
    )
    assert np.allclose(
        np.asarray(utilities_for_params(params)),
        np.asarray(utilities_without_density),
        rtol=1e-6,
        atol=1e-6,
    )

    objective_grad = jax.grad(lambda p: jnp.sum(utilities_for_params(p)))(params)
    per_cell_grads = [
        jax.grad(lambda p, idx=index: utilities_for_params(p)[idx])(params)
        for index in range(3)
    ]
    summed_per_cell = jax.tree_util.tree_map(lambda *items: sum(items), *per_cell_grads)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: jnp.allclose(a, b, rtol=1e-5, atol=1e-5), objective_grad, summed_per_cell)
    )

    loss, aux = promisys_twostep._joint_multicontext_infonce_loss(
        params,
        design_params,
        batch,
        design_key=design_key,
        log_prob_fn=log_prob_fn,
        bounds=bounds,
        weights=jnp.ones(3, dtype=jnp.float32) / 3.0,
        infonce_lambda=0.5,
        inner_samples=2,
    )
    design_grads = jax.grad(
        lambda current_design_params: promisys_twostep._joint_multicontext_infonce_loss(
            params,
            current_design_params,
            batch,
            design_key=design_key,
            log_prob_fn=log_prob_fn,
            bounds=bounds,
            weights=jnp.ones(3, dtype=jnp.float32) / 3.0,
            infonce_lambda=0.5,
            inner_samples=2,
        )[0]
    )(design_params)
    assert np.isfinite(float(loss))
    assert np.all(np.isfinite(np.asarray(aux["utilities"])))
    assert np.all(np.isfinite(np.asarray(design_grads["bmp4_norm_mu"])))
    assert np.all(np.isfinite(np.asarray(design_grads["bmp4_norm_log_std"])))


@pytest.mark.parametrize("module", [promisys_onestep, promisys_twostep])
def test_promisys_design_update_uses_plain_sgd(module) -> None:
    jnp = pytest.importorskip("jax.numpy")

    bounds = {"bmp4_norm_min": -2.0, "bmp4_norm_max": 2.0}
    design_params = {
        "bmp4_norm_mu": jnp.asarray([-1.0, 0.5], dtype=jnp.float32),
        "bmp4_norm_log_std": jnp.log(jnp.asarray([0.25, 0.5], dtype=jnp.float32)),
    }
    design_grads = {
        "bmp4_norm_mu": jnp.asarray([2.0, -4.0], dtype=jnp.float32),
        "bmp4_norm_log_std": jnp.asarray([0.6, -0.2], dtype=jnp.float32),
    }
    learning_rate = 0.1

    updated = module._apply_design_sgd_update(
        design_params,
        design_grads,
        bounds,
        learning_rate=learning_rate,
    )

    expected_mu = np.asarray(design_params["bmp4_norm_mu"]) - learning_rate * np.asarray(
        design_grads["bmp4_norm_mu"]
    )
    expected_log_std = np.asarray(design_params["bmp4_norm_log_std"])

    assert np.allclose(np.asarray(updated["bmp4_norm_mu"]), expected_mu, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(updated["bmp4_norm_log_std"]), expected_log_std, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("module", [promisys_onestep, promisys_twostep])
def test_promisys_design_update_plain_sgd_ignores_tiny_temperature_std(module) -> None:
    jnp = pytest.importorskip("jax.numpy")

    bounds = {"bmp4_norm_min": -2.0, "bmp4_norm_max": 2.0}
    design_params = {
        "bmp4_norm_mu": jnp.asarray([0.0], dtype=jnp.float32),
        "bmp4_norm_log_std": jnp.log(jnp.asarray([0.001], dtype=jnp.float32)),
    }
    design_grads = {
        "bmp4_norm_mu": jnp.asarray([1.0], dtype=jnp.float32),
        "bmp4_norm_log_std": jnp.asarray([0.0], dtype=jnp.float32),
    }

    updated = module._apply_design_sgd_update(
        design_params,
        design_grads,
        bounds,
        learning_rate=1.0,
    )

    assert np.asarray(updated["bmp4_norm_mu"])[0] == pytest.approx(-1.0, abs=1e-6)
    assert np.asarray(updated["bmp4_norm_log_std"])[0] == pytest.approx(
        np.asarray(design_params["bmp4_norm_log_std"])[0],
        abs=1e-6,
    )


def test_promisys_sequential_snaps_to_nearest_unused_design() -> None:
    bundle = load_bmp4_gradient_data(DEFAULT_DATA_PATH)
    joint = build_joint_bmp4_gradient_data(bundle, cell_lines=["NMuMG"])

    first = snap_to_nearest_unused_design(
        joint_data=joint,
        cell_line_index=0,
        proposed_bmp4_norm=float(joint.bmp4_conc_norm[0, 0]),
        proposed_log10_dose=float(math.log10(joint.bmp4_conc[0, 0])),
        used_designs=set(),
    )
    assert first["dose_index"] == 0

    second = snap_to_nearest_unused_design(
        joint_data=joint,
        cell_line_index=0,
        proposed_bmp4_norm=float(joint.bmp4_conc_norm[0, 0]),
        proposed_log10_dose=float(math.log10(joint.bmp4_conc[0, 0])),
        used_designs={("NMuMG", 0)},
    )
    assert second["dose_index"] == 1


def test_promisys_sequential_snap_ties_break_by_log_distance() -> None:
    joint = SimpleNamespace(
        cell_lines=("A",),
        bmp4_conc=np.asarray([[1.0, 10.0]], dtype=np.float32),
        bmp4_conc_norm=np.asarray([[0.0, 2.0]], dtype=np.float32),
    )

    snapped = snap_to_nearest_unused_design(
        joint_data=joint,
        cell_line_index=0,
        proposed_bmp4_norm=1.0,
        proposed_log10_dose=1.0,
        used_designs=set(),
    )

    assert snapped["dose_index"] == 1


def test_promisys_sequential_initial_prior_samples_support_default_and_literature() -> None:
    bundle = load_bmp4_gradient_data(DEFAULT_DATA_PATH)
    joint = build_joint_bmp4_gradient_data(bundle, cell_lines=["NMuMG", "BMPR2_KD"])
    literature_prior = AugmentedPrior(
        distributions={
            "kd_BMPR2": DistributionSpec(
                name="LogNormal",
                params={"loc": math.log(0.4), "scale": 0.05},
                source="literature",
                reasoning="test prior",
            )
        }
    )

    default_samples, default_prior = initialize_promisys_prior_samples(
        family_name="promisys_onestep",
        joint_data=joint,
        sample_count=3,
        prior_mode="default",
        seed=1,
    )
    literature_samples, literature_theta_prior = initialize_promisys_prior_samples(
        family_name="promisys_onestep",
        joint_data=joint,
        sample_count=3,
        prior_mode="literature",
        literature_prior=literature_prior,
        seed=1,
    )

    assert default_prior["mode"] == "default_loguniform"
    assert literature_theta_prior["mode"] == "expert_mapped"
    assert literature_theta_prior["mapped_parameter_count"] > 0
    assert default_samples["NMuMG"].shape == (3, promisys_onestep.THETA_DIM)
    assert literature_samples["BMPR2_KD"].shape == (3, promisys_onestep.THETA_DIM)


@pytest.mark.parametrize("family", ["hill", "multireceptor"])
def test_bmp4_gradient_runner_smoke(tmp_path: Path, family: str) -> None:
    pytest.importorskip("pyro")
    pytest.importorskip("matplotlib")

    llm = RecordingLLMClient(responder=_stub_literature_responder)
    problem_path = _write_stub_problem_bundle(tmp_path)
    output_dir = tmp_path / "artifacts"

    summary = run_bmp4_gradient_example(
        llm_client=llm,
        problem_path=problem_path,
        output_dir=output_dir,
        cell_lines=["NMuMG"],
        families=[family],
        fit_steps=4,
        fit_learning_rate=0.02,
        posterior_samples=8,
        eig_steps=3,
        eig_learning_rate=0.05,
        eig_outer_samples=4,
    )

    run_dir = output_dir / family / "NMuMG"
    assert len(summary["runs"]) == 1
    assert (run_dir / "literature_prior.json").exists()
    assert (run_dir / "fit_summary.json").exists()
    assert (run_dir / "posterior_samples.pt").exists()
    assert (run_dir / "posterior_predictive.pt").exists()
    assert (run_dir / "posterior_predictive.png").exists()
    assert (run_dir / "prior_posterior_comparison.png").exists()
    assert (run_dir / "prior_posterior_comparison_positive.png").exists()
    assert (run_dir / "eig_optimization_summary.json").exists()
    assert (run_dir / "eig_optimization.png").exists()
    assert (output_dir / "run_summary.json").exists()
    assert any(call["stage"] == "stage_b" for call in llm.calls)


def test_bmp4_gradient_runner_hierarchical_smoke(tmp_path: Path) -> None:
    pytest.importorskip("pyro")
    pytest.importorskip("matplotlib")

    llm = RecordingLLMClient(responder=_stub_literature_responder)
    problem_path = _write_stub_problem_bundle(tmp_path)
    output_dir = tmp_path / "artifacts"

    summary = run_bmp4_gradient_example(
        llm_client=llm,
        problem_path=problem_path,
        output_dir=output_dir,
        cell_lines=["NMuMG", "BMPR2_KD"],
        families=["multireceptor_hierarchical"],
        fit_steps=4,
        fit_learning_rate=0.02,
        posterior_samples=8,
        eig_steps=3,
        eig_learning_rate=0.05,
        eig_outer_samples=4,
    )

    run_dir = output_dir / "multireceptor_hierarchical" / "joint__NMuMG__BMPR2_KD"
    posterior_samples = torch.load(run_dir / "posterior_samples.pt", map_location="cpu")
    posterior_predictive = torch.load(run_dir / "posterior_predictive.pt", map_location="cpu")
    fit_summary = json.loads((run_dir / "fit_summary.json").read_text(encoding="utf-8"))
    eig_summary = json.loads((run_dir / "eig_optimization_summary.json").read_text(encoding="utf-8"))

    assert len(summary["runs"]) == 1
    assert (run_dir / "literature_prior.json").exists()
    assert (run_dir / "posterior_predictive.pt").exists()
    assert (run_dir / "posterior_predictive.png").exists()
    assert (run_dir / "prior_posterior_comparison.png").exists()
    assert (run_dir / "prior_posterior_comparison_positive.png").exists()
    assert (run_dir / "eig_optimization.png").exists()
    assert fit_summary["q_obs_mode"] == "joint_qpcr_measurement_layer"
    assert "log_R" in posterior_samples
    assert "log_kd" in posterior_samples
    assert "q_obs" not in posterior_samples
    assert posterior_samples["log_R"].shape[-2:] == (2, 5)
    assert torch.all(posterior_predictive["predictive_y"] > 0)
    assert eig_summary["design_type"] == "cell_line_selector_plus_bmp4_dose"
    assert eig_summary["best_cell_line"] in {"NMuMG", "BMPR2_KD"}
    assert len(eig_summary["best_selector_probs"]) == 2
    assert eig_summary["best_dose"] > 0.0
    assert eig_summary["best_log10_dose"] is not None
    assert eig_summary["optimization_bounds"]["log10_dose_min"] < eig_summary["optimization_bounds"]["log10_dose_max"]
    assert eig_summary["optimization_bounds"]["dose_min"] <= eig_summary["best_dose"] <= eig_summary["optimization_bounds"]["dose_max"]
    assert len(eig_summary["history"][0]["selector_probs"]) == 2
    assert "log10_dose" in eig_summary["history"][0]


def test_bmp4_gradient_runner_promisys_onestep_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("jax")
    pytest.importorskip("haiku")
    pytest.importorskip("optax")

    def fake_simulate_promisys_onestep_raw(
        *,
        bmp4_concentrations,
        receptors,
        theta_norm,
        receptor_noise_log_sd=0.0,
        rng=None,
        lfiax_root=None,
    ):
        _ = receptor_noise_log_sd, rng, lfiax_root
        doses = np.asarray(bmp4_concentrations, dtype=np.float32).reshape(-1)
        theta = np.asarray(theta_norm, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta[None, :]
        receptors_arr = np.asarray(receptors, dtype=np.float32).reshape(-1)
        theta_signal = 0.15 + 0.03 * theta[:, :6].mean(axis=1, keepdims=True)
        receptor_signal = 0.01 * float(np.mean(receptors_arr))
        dose_signal = np.log1p(np.maximum(doses, 0.0))[None, :] * 0.05
        return np.maximum(theta_signal + receptor_signal + dose_signal, 1e-5).astype("float32")

    def fake_simulate_promisys_onestep_paired_raw(
        *,
        bmp4_concentrations,
        receptors,
        theta_norm,
        receptor_noise_log_sd=0.0,
        rng=None,
        lfiax_root=None,
    ):
        _ = receptor_noise_log_sd, rng, lfiax_root
        doses = np.asarray(bmp4_concentrations, dtype=np.float32).reshape(-1)
        theta = np.asarray(theta_norm, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta[None, :]
        receptors_arr = np.asarray(receptors, dtype=np.float32).reshape(-1)
        theta_signal = 0.15 + 0.03 * theta[:, :6].mean(axis=1)
        receptor_signal = 0.01 * float(np.mean(receptors_arr))
        dose_signal = np.log1p(np.maximum(doses, 0.0)) * 0.05
        return np.maximum(theta_signal + receptor_signal + dose_signal, 1e-5).astype("float32")

    monkeypatch.setattr(promisys_onestep, "_require_promisys", lambda lfiax_root: object())
    monkeypatch.setattr(promisys_onestep, "simulate_promisys_onestep_raw", fake_simulate_promisys_onestep_raw)
    monkeypatch.setattr(promisys_onestep, "simulate_promisys_onestep_paired_raw", fake_simulate_promisys_onestep_paired_raw)

    llm = RecordingLLMClient(responder=_stub_literature_responder)
    problem_path = _write_stub_problem_bundle(tmp_path)
    output_dir = tmp_path / "artifacts"
    normalizer_dir = _write_synthetic_normalizer_sources(tmp_path)
    prior_path = tmp_path / "onestep_expert_prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "family": "bmp4_expert",
                "priors": {
                    "kd_ACVR2A": {
                        "distribution": "LogNormal",
                        "params": {"loc": math.log(0.8), "scale": 0.05},
                        "reasoning": "test prior",
                    },
                    "kd_ACVR2B": {
                        "distribution": "LogNormal",
                        "params": {"loc": math.log(1.6), "scale": 0.05},
                        "reasoning": "test prior",
                    },
                    "kd_BMPR2": {
                        "distribution": "LogNormal",
                        "params": {"loc": math.log(0.4), "scale": 0.05},
                        "reasoning": "test prior",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = run_bmp4_gradient_example(
        llm_client=llm,
        problem_path=problem_path,
        output_dir=output_dir,
        cell_lines=["NMuMG", "BMPR2_KD"],
        families=["promisys_onestep"],
        literature_prior_jsons=[str(prior_path)],
        fit_steps=1,
        fit_learning_rate=1e-3,
        posterior_samples=4,
        eig_steps=1,
        eig_outer_samples=2,
        eig_inner_samples=1,
        infonce_lambda=0.5,
        normalizer_data_dir=normalizer_dir,
        snpe_steps=1,
        snpe_simulations=2,
        snpe_learning_rate=1e-3,
        mcmc_warmup=1,
        mcmc_samples=2,
        promisys_hyperparams=_tiny_promisys_hyperparams(),
    )

    run_dir = output_dir / "promisys_onestep" / "onestep_expert_prior" / "joint__NMuMG__BMPR2_KD"
    eig_summary = json.loads((run_dir / "eig_optimization_summary.json").read_text(encoding="utf-8"))
    fit_summary = json.loads((run_dir / "fit_summary.json").read_text(encoding="utf-8"))
    checkpoint = pickle.loads((run_dir / "likelihood_checkpoint.pkl").read_bytes())

    assert len(summary["runs"]) == 1
    assert (run_dir / "snpe_posterior_samples.pt").exists()
    assert (run_dir / "posterior_samples.pt").exists()
    assert (run_dir / "likelihood_checkpoint.pkl").exists()
    assert (run_dir / "mcmc_posterior_samples.pt").exists()
    assert (run_dir / "posterior_predictive.png").exists()
    assert fit_summary["candidate_name"] == "promisys_onestep_lfiax"
    assert summary["runs"][0]["literature_prior_source"] == str(prior_path)
    assert fit_summary["simulation_summary"]["theta_raw_prior"]["mode"] == "expert_mapped"
    assert fit_summary["simulation_summary"]["theta_raw_prior"]["mapped_parameter_count"] == 6
    assert eig_summary["design_type"] == "cell_line_selector_plus_bmp4_dose"
    assert eig_summary["objective"] == "fixed_posterior_multi_context_lfiax"
    assert eig_summary["fit_steps_used_for_joint_boed"] == 1
    assert len(eig_summary["history"]) == 1
    assert "per_cell_utilities" in eig_summary["history"][0]
    assert np.allclose(eig_summary["history"][0]["selector_probs"], [0.5, 0.5])
    assert "selector_gradients" in eig_summary["gradient_diagnostics"][0]
    assert eig_summary["prior_eig_baseline"]["mapped_parameter_count"] == 6
    assert math.isfinite(eig_summary["prior_eig_baseline"]["mean"])
    assert "estimated_total_eig" in eig_summary["history"][0]
    assert "per_cell_bmp4_norm_mu" in eig_summary["history"][0]
    assert "per_cell_bmp4_norm_std" in eig_summary["history"][0]
    assert "per_cell_dose_mu" in eig_summary["history"][0]
    assert "per_cell_dose_std" in eig_summary["history"][0]
    assert eig_summary["best_bmp4_norm_mu"] is not None
    assert eig_summary["best_bmp4_norm_std"] is not None
    assert eig_summary["optimization_bounds"]["bmp4_norm_min"] < eig_summary["optimization_bounds"]["bmp4_norm_max"]
    assert (
        eig_summary["optimization_bounds"]["bmp4_norm_min"]
        <= eig_summary["best_bmp4_norm_mu"]
        <= eig_summary["optimization_bounds"]["bmp4_norm_max"]
    )
    assert eig_summary["gradient_diagnostics"]
    assert "mu_gradients" in eig_summary["gradient_diagnostics"][0]
    assert "std_gradients" in eig_summary["gradient_diagnostics"][0]
    assert "utility_design_jacobian" not in eig_summary["gradient_diagnostics"][0]
    assert "final_next_experiment" in eig_summary
    assert "dose_mu" in eig_summary["final_next_experiment"]
    assert "dose_std" in eig_summary["final_next_experiment"]
    assert eig_summary["best_cell_line"] in {"NMuMG", "BMPR2_KD"}
    assert "bmp4_norm_mu" in checkpoint["jax_likelihood"]
    assert "bmp4_norm_std" in checkpoint["jax_likelihood"]
    assert "dose_mu" in checkpoint["jax_likelihood"]
    assert "dose_std" in checkpoint["jax_likelihood"]
    assert "selector_probs" in checkpoint["jax_likelihood"]
    assert "bmp4_norm_designs" not in checkpoint["jax_likelihood"]
    assert checkpoint["metadata"]["theta_names"][-1] == promisys_onestep.OBSERVATION_NOISE_PARAMETER_NAME
    assert checkpoint["metadata"]["snpe_theta_prior"]["mode"] == "expert_mapped"
    assert checkpoint["metadata"]["observation_noise_prior"]["parameter_name"] == promisys_onestep.OBSERVATION_NOISE_PARAMETER_NAME
    assert fit_summary["promisys_hyperparams"]["posterior_net"]["hidden_dim"] == 16
    assert fit_summary["mcmc_sample_count_by_cell"] == {"NMuMG": 2, "BMPR2_KD": 2}


def test_bmp4_gradient_runner_promisys_twostep_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("jax")
    pytest.importorskip("haiku")
    pytest.importorskip("optax")

    def fake_simulate_promisys_twostep_raw(
        *,
        bmp4_concentrations,
        receptors,
        theta_norm,
        receptor_noise_log_sd=0.0,
        rng=None,
        lfiax_root=None,
    ):
        _ = receptor_noise_log_sd, rng, lfiax_root
        doses = np.asarray(bmp4_concentrations, dtype=np.float32).reshape(-1)
        theta = np.asarray(theta_norm, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta[None, :]
        receptors_arr = np.asarray(receptors, dtype=np.float32).reshape(-1)
        theta_signal = 0.15 + 0.02 * theta[:, :8].mean(axis=1, keepdims=True)
        receptor_signal = 0.01 * float(np.mean(receptors_arr))
        dose_signal = np.log1p(np.maximum(doses, 0.0))[None, :] * 0.05
        return np.maximum(theta_signal + receptor_signal + dose_signal, 1e-5).astype("float32")

    def fake_simulate_promisys_twostep_paired_raw(
        *,
        bmp4_concentrations,
        receptors,
        theta_norm,
        receptor_noise_log_sd=0.0,
        rng=None,
        lfiax_root=None,
    ):
        _ = receptor_noise_log_sd, rng, lfiax_root
        doses = np.asarray(bmp4_concentrations, dtype=np.float32).reshape(-1)
        theta = np.asarray(theta_norm, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta[None, :]
        receptors_arr = np.asarray(receptors, dtype=np.float32).reshape(-1)
        theta_signal = 0.15 + 0.02 * theta[:, :8].mean(axis=1)
        receptor_signal = 0.01 * float(np.mean(receptors_arr))
        dose_signal = np.log1p(np.maximum(doses, 0.0)) * 0.05
        return np.maximum(theta_signal + receptor_signal + dose_signal, 1e-5).astype("float32")

    monkeypatch.setattr(promisys_twostep, "_require_promisys", lambda lfiax_root: object())
    monkeypatch.setattr(promisys_twostep, "simulate_promisys_twostep_raw", fake_simulate_promisys_twostep_raw)
    monkeypatch.setattr(promisys_twostep, "simulate_promisys_twostep_paired_raw", fake_simulate_promisys_twostep_paired_raw)

    llm = RecordingLLMClient(responder=_stub_literature_responder)
    problem_path = _write_stub_problem_bundle(tmp_path)
    output_dir = tmp_path / "artifacts"
    normalizer_dir = _write_synthetic_normalizer_sources(tmp_path)
    prior_path = tmp_path / "twostep_expert_prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "family": "bmp4_expert",
                "priors": {
                    "kd_BMPR1A": {
                        "distribution": "LogNormal",
                        "params": {"loc": math.log(80.0), "scale": 0.05},
                        "reasoning": "test prior",
                    },
                    "kd_ACVR2A": {
                        "distribution": "LogNormal",
                        "params": {"loc": math.log(0.8), "scale": 0.05},
                        "reasoning": "test prior",
                    },
                    "kd_ACVR2B": {
                        "distribution": "LogNormal",
                        "params": {"loc": math.log(1.6), "scale": 0.05},
                        "reasoning": "test prior",
                    },
                    "kd_BMPR2": {
                        "distribution": "LogNormal",
                        "params": {"loc": math.log(0.4), "scale": 0.05},
                        "reasoning": "test prior",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = run_bmp4_gradient_example(
        llm_client=llm,
        problem_path=problem_path,
        output_dir=output_dir,
        cell_lines=["NMuMG", "BMPR2_KD"],
        families=["promisys_twostep"],
        literature_prior_jsons=[str(prior_path)],
        fit_steps=1,
        fit_learning_rate=1e-3,
        posterior_samples=4,
        eig_steps=1,
        eig_outer_samples=2,
        eig_inner_samples=1,
        infonce_lambda=0.5,
        normalizer_data_dir=normalizer_dir,
        snpe_steps=1,
        snpe_simulations=2,
        snpe_learning_rate=1e-3,
        mcmc_warmup=1,
        mcmc_samples=2,
        promisys_hyperparams=_tiny_promisys_hyperparams(),
    )

    run_dir = output_dir / "promisys_twostep" / "twostep_expert_prior" / "joint__NMuMG__BMPR2_KD"
    eig_summary = json.loads((run_dir / "eig_optimization_summary.json").read_text(encoding="utf-8"))
    fit_summary = json.loads((run_dir / "fit_summary.json").read_text(encoding="utf-8"))
    checkpoint = pickle.loads((run_dir / "likelihood_checkpoint.pkl").read_bytes())

    assert len(summary["runs"]) == 1
    assert (run_dir / "snpe_posterior_samples.pt").exists()
    assert (run_dir / "posterior_samples.pt").exists()
    assert (run_dir / "likelihood_checkpoint.pkl").exists()
    assert (run_dir / "mcmc_posterior_samples.pt").exists()
    assert (run_dir / "posterior_predictive.png").exists()
    assert fit_summary["candidate_name"] == "promisys_twostep_lfiax"
    assert summary["runs"][0]["literature_prior_source"] == str(prior_path)
    assert fit_summary["simulation_summary"]["theta_raw_prior"]["mode"] == "expert_mapped"
    assert fit_summary["simulation_summary"]["theta_raw_prior"]["mapped_parameter_count"] == 7
    assert eig_summary["design_type"] == "cell_line_selector_plus_bmp4_dose"
    assert eig_summary["objective"] == "fixed_posterior_multi_context_lfiax"
    assert eig_summary["fit_steps_used_for_joint_boed"] == 1
    assert len(eig_summary["history"]) == 1
    assert "per_cell_utilities" in eig_summary["history"][0]
    assert np.allclose(eig_summary["history"][0]["selector_probs"], [0.5, 0.5])
    assert "selector_gradients" in eig_summary["gradient_diagnostics"][0]
    assert eig_summary["prior_eig_baseline"]["mapped_parameter_count"] == 7
    assert math.isfinite(eig_summary["prior_eig_baseline"]["mean"])
    assert "estimated_total_eig" in eig_summary["history"][0]
    assert "per_cell_bmp4_norm_mu" in eig_summary["history"][0]
    assert "per_cell_bmp4_norm_std" in eig_summary["history"][0]
    assert "per_cell_dose_mu" in eig_summary["history"][0]
    assert "per_cell_dose_std" in eig_summary["history"][0]
    assert eig_summary["best_bmp4_norm_mu"] is not None
    assert eig_summary["best_bmp4_norm_std"] is not None
    assert eig_summary["optimization_bounds"]["bmp4_norm_min"] < eig_summary["optimization_bounds"]["bmp4_norm_max"]
    assert (
        eig_summary["optimization_bounds"]["bmp4_norm_min"]
        <= eig_summary["best_bmp4_norm_mu"]
        <= eig_summary["optimization_bounds"]["bmp4_norm_max"]
    )
    assert eig_summary["gradient_diagnostics"]
    assert "mu_gradients" in eig_summary["gradient_diagnostics"][0]
    assert "std_gradients" in eig_summary["gradient_diagnostics"][0]
    assert "final_next_experiment" in eig_summary
    assert "dose_mu" in eig_summary["final_next_experiment"]
    assert "dose_std" in eig_summary["final_next_experiment"]
    assert eig_summary["best_cell_line"] in {"NMuMG", "BMPR2_KD"}
    assert "bmp4_norm_mu" in checkpoint["jax_likelihood"]
    assert "bmp4_norm_std" in checkpoint["jax_likelihood"]
    assert "dose_mu" in checkpoint["jax_likelihood"]
    assert "dose_std" in checkpoint["jax_likelihood"]
    assert "selector_probs" in checkpoint["jax_likelihood"]
    assert checkpoint["metadata"]["theta_names"][-1] == promisys_twostep.OBSERVATION_NOISE_PARAMETER_NAME
    assert checkpoint["metadata"]["dimer_complex_names"] == list(promisys_twostep.DIMER_COMPLEX_NAMES)
    assert checkpoint["metadata"]["snpe_theta_prior"]["mode"] == "expert_mapped"
    assert checkpoint["metadata"]["observation_noise_prior"]["parameter_name"] == promisys_twostep.OBSERVATION_NOISE_PARAMETER_NAME
    assert fit_summary["promisys_hyperparams"]["posterior_net"]["hidden_dim"] == 16
    assert fit_summary["mcmc_sample_count_by_cell"] == {"NMuMG": 2, "BMPR2_KD": 2}


@pytest.mark.parametrize(
    ("family", "module", "raw_name", "paired_name"),
    [
        (
            "promisys_onestep",
            promisys_onestep,
            "simulate_promisys_onestep_raw",
            "simulate_promisys_onestep_paired_raw",
        ),
        (
            "promisys_twostep",
            promisys_twostep,
            "simulate_promisys_twostep_raw",
            "simulate_promisys_twostep_paired_raw",
        ),
    ],
)
def test_promisys_sequential_workflow_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    module: Any,
    raw_name: str,
    paired_name: str,
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("jax")
    pytest.importorskip("haiku")
    pytest.importorskip("optax")

    def fake_raw(
        *,
        bmp4_concentrations,
        receptors,
        theta_norm,
        receptor_noise_log_sd=0.0,
        rng=None,
        lfiax_root=None,
    ):
        _ = receptor_noise_log_sd, rng, lfiax_root
        doses = np.asarray(bmp4_concentrations, dtype=np.float32).reshape(-1)
        theta = np.asarray(theta_norm, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta[None, :]
        receptor_signal = 0.01 * float(np.mean(np.asarray(receptors, dtype=np.float32)))
        theta_signal = 0.12 + 0.02 * theta[:, : min(6, theta.shape[1])].mean(axis=1, keepdims=True)
        dose_signal = np.log1p(np.maximum(doses, 0.0))[None, :] * 0.04
        return np.maximum(theta_signal + receptor_signal + dose_signal, 1e-5).astype("float32")

    def fake_paired(
        *,
        bmp4_concentrations,
        receptors,
        theta_norm,
        receptor_noise_log_sd=0.0,
        rng=None,
        lfiax_root=None,
    ):
        _ = receptor_noise_log_sd, rng, lfiax_root
        doses = np.asarray(bmp4_concentrations, dtype=np.float32).reshape(-1)
        theta = np.asarray(theta_norm, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta[None, :]
        receptor_signal = 0.01 * float(np.mean(np.asarray(receptors, dtype=np.float32)))
        theta_signal = 0.12 + 0.02 * theta[:, : min(6, theta.shape[1])].mean(axis=1)
        dose_signal = np.log1p(np.maximum(doses, 0.0)) * 0.04
        return np.maximum(theta_signal + receptor_signal + dose_signal, 1e-5).astype("float32")

    monkeypatch.setattr(module, "_require_promisys", lambda lfiax_root: object())
    monkeypatch.setattr(module, raw_name, fake_raw)
    monkeypatch.setattr(module, paired_name, fake_paired)

    bundle = load_bmp4_gradient_data(DEFAULT_DATA_PATH)
    joint = build_joint_bmp4_gradient_data(bundle, cell_lines=["NMuMG", "BMPR2_KD"])
    normalizer_dir = _write_synthetic_normalizer_sources(tmp_path)

    summary = run_promisys_sequential_workflow(
        family_name=family,
        joint_data=joint,
        run_dir=tmp_path / "sequential",
        prior_mode="default",
        normalizer_data_dir=normalizer_dir,
        likelihood_steps=1,
        likelihood_learning_rate=1e-3,
        posterior_sample_count=4,
        eig_steps=1,
        eig_outer_samples=2,
        eig_inner_samples=1,
        eig_learning_rate=0.05,
        mcmc_warmup=1,
        mcmc_samples=2,
        rounds=1,
        batch_size=2,
        promisys_hyperparams=_tiny_promisys_hyperparams(),
    )

    run_dir = Path(summary["run_dir"])
    trace = json.loads((run_dir / "sequential_trace.json").read_text(encoding="utf-8"))["trace"]
    used = {
        (item["snapped_design"]["cell_line"], item["snapped_design"]["dose_index"])
        for item in trace
    }
    fit_summary = json.loads((run_dir / "fit_summary.json").read_text(encoding="utf-8"))

    assert summary["acquisition_count"] == 2
    assert len(trace) == 2
    assert len(used) == 2
    assert (run_dir / "step_001" / "mcmc_posterior_samples.pt").exists()
    assert (run_dir / "step_002" / "mcmc_posterior_samples.pt").exists()
    assert (run_dir / "posterior_samples.pt").exists()
    assert (run_dir / "posterior_predictive.png").exists()
    assert fit_summary["candidate_name"] == f"{family}_sequential_lfiax"
    assert fit_summary["acquisition_count"] == 2
    assert sum(fit_summary["collected_count_by_cell"].values()) == 2


def test_bmp4_comparison_runner_writes_baseline_and_sequential_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_baseline(**kwargs):
        assert kwargs["fit_steps"] == 50
        assert kwargs["early_stopping_patience"] == 10
        assert kwargs["promisys_hyperparams"].objective.early_stopping_patience == 10
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"runs": [{"run_dir": str(output_dir / "baseline_leaf")}], "output_dir": str(output_dir)}

    def fake_sequential(**kwargs):
        assert kwargs["likelihood_steps"] == 50
        assert kwargs["early_stopping_patience"] == 10
        assert kwargs["promisys_hyperparams"].objective.early_stopping_min_delta == 0.0
        run_dir = Path(kwargs["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_path = run_dir / "sequential_trace.json"
        trace_path.write_text(json.dumps({"trace": []}), encoding="utf-8")
        return {
            "run_dir": str(run_dir),
            "trace_path": str(trace_path),
            "acquisition_count": int(kwargs["rounds"]) * int(kwargs["batch_size"]),
        }

    monkeypatch.setattr(bmp4_gradient_comparison, "run_bmp4_gradient_example", fake_baseline)
    monkeypatch.setattr(bmp4_gradient_comparison, "run_promisys_sequential_workflow", fake_sequential)
    hyperparams_path = tmp_path / "promisys_hyperparams.json"
    hyperparams_path.write_text(
        json.dumps(
            {
                "objective": {
                    "fit_steps": 50,
                    "early_stopping_patience": 10,
                    "early_stopping_min_delta": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

    summary = bmp4_gradient_comparison.run_bmp4_promisys_comparison(
        family="promisys_onestep",
        output_dir=tmp_path,
        run_label="comparison_test",
        cell_lines=["NMuMG"],
        prior_mode="default",
        promisys_hyperparams_json=hyperparams_path,
        rounds=1,
        batch_size=1,
    )

    root = tmp_path / "comparison_test"
    assert summary["baseline_enabled"] is True
    assert "default" in summary["baseline"]
    assert "default" in summary["sequential"]
    assert (root / "comparison_summary.json").exists()
    assert (root / "sequential_default" / "comparison_summary.json").exists()


def test_bmp4_comparison_runner_uses_default_literature_prior_for_both(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_path = tmp_path / "literature_prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "family": "multireceptor_hierarchical",
                "priors": {
                    "kd_BMPR2": {
                        "distribution": "LogNormal",
                        "params": {"loc": -1.0, "scale": 0.2},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bmp4_gradient_comparison, "DEFAULT_LITERATURE_PRIOR", prior_path)
    monkeypatch.setattr(bmp4_gradient_comparison, "run_bmp4_gradient_example", lambda **kwargs: {})

    def fake_sequential(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_path = run_dir / "sequential_trace.json"
        trace_path.write_text(json.dumps({"trace": []}), encoding="utf-8")
        return {"run_dir": str(run_dir), "trace_path": str(trace_path), "acquisition_count": 0}

    monkeypatch.setattr(bmp4_gradient_comparison, "run_promisys_sequential_workflow", fake_sequential)

    summary = bmp4_gradient_comparison.run_bmp4_promisys_comparison(
        family="promisys_twostep",
        output_dir=tmp_path,
        run_label="default_literature",
        cell_lines=["NMuMG"],
        prior_mode="both",
        rounds=0,
    )

    assert summary["literature_prior_json"] == str(prior_path)
    assert summary["prior_modes_run"] == ["default", "literature"]


def test_sequential_analysis_writes_metrics_and_uses_cumulative_designs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")

    def fake_simulate_twostep_raw(
        *,
        bmp4_concentrations,
        receptors,
        theta_norm,
        receptor_noise_log_sd=0.0,
        rng=None,
        lfiax_root=None,
    ):
        _ = receptors, receptor_noise_log_sd, rng, lfiax_root
        doses = np.asarray(bmp4_concentrations, dtype=np.float32).reshape(-1)
        theta = np.asarray(theta_norm, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta[None, :]
        return (
            0.12
            + 0.02 * theta[:, :4].mean(axis=1, keepdims=True)
            + 0.04 * np.log1p(np.maximum(doses, 0.0))[None, :]
        ).astype("float32")

    monkeypatch.setattr(promisys_twostep, "simulate_promisys_twostep_raw", fake_simulate_twostep_raw)
    run_dir, normalizer_dir = _write_synthetic_sequential_analysis_run(tmp_path, "default")

    result = analyze_sequential_run(
        run_dir,
        normalizer_data_dir=normalizer_dir,
        max_posterior_draws=3,
    )
    metrics = result["metrics"]

    assert len(metrics) == 2
    assert metrics[0]["n_acquired_points"] == 1
    assert metrics[1]["n_acquired_points"] == 2
    assert math.isfinite(metrics[0]["median_predictive_abs_error_norm"])
    assert "per_cell_utility_NMuMG" in metrics[0]
    assert (run_dir / "sequential_metrics.json").exists()
    assert (run_dir / "sequential_metrics.tsv").exists()
    assert (run_dir / "sequential_diagnostics.png").exists()


def test_sequential_analysis_writes_comparison_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")

    def fake_simulate_twostep_raw(
        *,
        bmp4_concentrations,
        receptors,
        theta_norm,
        receptor_noise_log_sd=0.0,
        rng=None,
        lfiax_root=None,
    ):
        _ = receptors, receptor_noise_log_sd, rng, lfiax_root
        doses = np.asarray(bmp4_concentrations, dtype=np.float32).reshape(-1)
        theta = np.asarray(theta_norm, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta[None, :]
        return (0.1 + 0.03 * np.log1p(np.maximum(doses, 0.0))[None, :] + 0.0 * theta[:, :1]).astype("float32")

    monkeypatch.setattr(promisys_twostep, "simulate_promisys_twostep_raw", fake_simulate_twostep_raw)
    comparison_dir = tmp_path / "comparison"
    _, normalizer_dir = _write_synthetic_sequential_analysis_run(comparison_dir, "default")
    _write_synthetic_sequential_analysis_run(comparison_dir, "literature", normalizer_dir=normalizer_dir)

    result = analyze_comparison_run(
        comparison_dir,
        normalizer_data_dir=normalizer_dir,
        max_posterior_draws=3,
    )

    assert result["run_count"] == 2
    assert (comparison_dir / "comparison_metrics.json").exists()
    assert (comparison_dir / "comparison_diagnostics.png").exists()
    assert (comparison_dir / "sequential_default" / "sequential_metrics.json").exists()
    assert (comparison_dir / "sequential_literature" / "sequential_metrics.json").exists()


def test_multireceptor_runner_uses_qpcr_prior_only(tmp_path: Path) -> None:
    pytest.importorskip("pyro")
    pytest.importorskip("matplotlib")

    llm = RecordingLLMClient(responder=_stub_literature_responder)
    problem_path = _write_stub_problem_bundle(tmp_path)
    output_dir = tmp_path / "artifacts"

    run_bmp4_gradient_example(
        llm_client=llm,
        problem_path=problem_path,
        output_dir=output_dir,
        cell_lines=["NMuMG"],
        families=["multireceptor"],
        fit_steps=4,
        fit_learning_rate=0.02,
        posterior_samples=8,
        eig_steps=3,
        eig_learning_rate=0.05,
        eig_outer_samples=4,
    )

    run_dir = output_dir / "multireceptor" / "NMuMG"
    fit_summary = json.loads((run_dir / "fit_summary.json").read_text(encoding="utf-8"))
    translated_sites = fit_summary["translated_prior"]["sites"]
    posterior_samples = torch.load(run_dir / "posterior_samples.pt", map_location="cpu")

    abundance_keys = [key for key in translated_sites if key.startswith("abundance_")]
    assert len(abundance_keys) == 5
    assert all(translated_sites[key]["source"] == "cell_line_qpcr" for key in abundance_keys)
    assert all(translated_sites[key]["distribution"] == "TruncatedLogNormal" for key in abundance_keys)
    assert all(translated_sites[key]["params"]["high"] == 5.0 for key in abundance_keys)
    assert "Rs" not in posterior_samples
    assert not any("qpcr" in key.lower() for key in posterior_samples)
    assert set(key for key in posterior_samples if key.startswith("abundance_")) == set(abundance_keys)


def test_bmp4_gradient_runner_supports_variational_boed_for_hill(tmp_path: Path) -> None:
    pytest.importorskip("pyro")
    pytest.importorskip("matplotlib")

    llm = RecordingLLMClient(responder=_stub_literature_responder)
    problem_path = _write_stub_problem_bundle(tmp_path)
    output_dir = tmp_path / "artifacts"

    summary = run_bmp4_gradient_example(
        llm_client=llm,
        problem_path=problem_path,
        output_dir=output_dir,
        cell_lines=["NMuMG"],
        families=["hill"],
        fit_steps=4,
        fit_learning_rate=0.02,
        posterior_samples=8,
        eig_estimator="posterior_eig",
        eig_steps=1,
        eig_guide_steps=2,
        eig_learning_rate=0.02,
        eig_guide_learning_rate=0.02,
        eig_outer_samples=2,
    )

    run_dir = output_dir / "hill" / "NMuMG"
    eig_summary = json.loads((run_dir / "eig_optimization_summary.json").read_text(encoding="utf-8"))

    assert summary["eig_estimator"] == "posterior_eig"
    assert eig_summary["backend"] == "pyro"
    assert eig_summary["estimator"] == "posterior_eig"
    assert eig_summary["best_dose"] > 0.0
    assert eig_summary["best_log10_dose"] is not None
    assert eig_summary["guide_ref"] is not None
    assert eig_summary["guide_training_steps"] == 2


def test_bmp4_gradient_runner_supports_variational_boed_for_hierarchical_family(tmp_path: Path) -> None:
    pytest.importorskip("pyro")
    pytest.importorskip("matplotlib")

    llm = RecordingLLMClient(responder=_stub_literature_responder)
    problem_path = _write_stub_problem_bundle(tmp_path)
    output_dir = tmp_path / "artifacts"

    summary = run_bmp4_gradient_example(
        llm_client=llm,
        problem_path=problem_path,
        output_dir=output_dir,
        cell_lines=["NMuMG", "BMPR2_KD"],
        families=["multireceptor_hierarchical"],
        fit_steps=4,
        fit_learning_rate=0.02,
        posterior_samples=8,
        eig_estimator="posterior_eig",
        eig_steps=1,
        eig_guide_steps=2,
        eig_learning_rate=0.02,
        eig_guide_learning_rate=0.02,
        eig_outer_samples=2,
    )

    run_dir = output_dir / "multireceptor_hierarchical" / "joint__NMuMG__BMPR2_KD"
    eig_summary = json.loads((run_dir / "eig_optimization_summary.json").read_text(encoding="utf-8"))

    assert summary["eig_estimator"] == "posterior_eig"
    assert eig_summary["backend"] == "pyro"
    assert eig_summary["estimator"] == "posterior_eig"
    assert eig_summary["best_cell_line"] in {"NMuMG", "BMPR2_KD"}
    assert len(eig_summary["best_selector_probs"]) == 2
    assert eig_summary["best_dose"] > 0.0
    assert eig_summary["guide_ref"] is not None
    assert eig_summary["guide_training_steps"] == 2


def test_bmp4_gradient_runner_reuses_saved_literature_prior_json(tmp_path: Path) -> None:
    pytest.importorskip("pyro")
    pytest.importorskip("matplotlib")

    llm = RecordingLLMClient(responder=_stub_literature_responder)
    problem_path = _write_stub_problem_bundle(tmp_path)
    output_dir = tmp_path / "artifacts"
    prior_path = tmp_path / "hill_literature_prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "family": "hill",
                "corpus_scope": "local_bmp4_corpus_only",
                "parameter_order": ["bottom", "top", "ec50", "hill_n", "sigma"],
                "priors": {
                    name: {
                        "distribution": spec["distribution"],
                        "params": spec["params"],
                        "reasoning": f"saved codex prior for {name}",
                        "cited_papers": ["saved-prior-note"],
                        "fallback": False,
                    }
                    for name, spec in _PRIOR_LIBRARY.items()
                    if name in {"bottom", "top", "ec50", "hill_n", "sigma"}
                },
                "notes": ["reused saved literature prior"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = run_bmp4_gradient_example(
        llm_client=llm,
        problem_path=problem_path,
        output_dir=output_dir,
        cell_lines=["NMuMG"],
        families=["hill"],
        literature_prior_jsons=[str(prior_path)],
        fit_steps=4,
        fit_learning_rate=0.02,
        posterior_samples=8,
        eig_steps=3,
        eig_learning_rate=0.05,
        eig_outer_samples=4,
    )

    run_dir = output_dir / "hill" / "NMuMG"
    fit_summary = json.loads((run_dir / "fit_summary.json").read_text(encoding="utf-8"))

    assert summary["runs"][0]["literature_prior_source"] == str(prior_path)
    assert llm.calls == []
    assert fit_summary["translated_prior"]["sites"]["bottom"]["source"] == "literature_json"
    assert fit_summary["translated_prior"]["sites"]["ec50"]["source"] == "literature_json"


def test_bmp4_gradient_runner_allows_shared_saved_prior_across_families(tmp_path: Path) -> None:
    pytest.importorskip("pyro")
    pytest.importorskip("matplotlib")

    llm = RecordingLLMClient(responder=_stub_literature_responder)
    problem_path = _write_stub_problem_bundle(tmp_path)
    output_dir = tmp_path / "artifacts"
    prior_path = tmp_path / "shared_literature_prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "family": "hill",
                "corpus_scope": "local_bmp4_corpus_only",
                "parameter_order": list(_PRIOR_LIBRARY),
                "priors": {
                    name: {
                        "distribution": spec["distribution"],
                        "params": spec["params"],
                        "reasoning": f"shared saved prior for {name}",
                        "cited_papers": ["shared-prior-note"],
                        "fallback": False,
                    }
                    for name, spec in _PRIOR_LIBRARY.items()
                },
                "notes": ["shared across conceptual families"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = run_bmp4_gradient_example(
        llm_client=llm,
        problem_path=problem_path,
        output_dir=output_dir,
        cell_lines=["NMuMG"],
        families=["multireceptor"],
        literature_prior_jsons=[str(prior_path)],
        fit_steps=4,
        fit_learning_rate=0.02,
        posterior_samples=8,
        eig_steps=3,
        eig_learning_rate=0.05,
        eig_outer_samples=4,
    )

    run_dir = output_dir / "multireceptor" / "NMuMG"
    literature_payload = json.loads((run_dir / "literature_prior.json").read_text(encoding="utf-8"))

    assert summary["runs"][0]["literature_prior_source"] == str(prior_path)
    assert llm.calls == []
    assert literature_payload["literature_report"]["family_reused_across_models"] is True
    assert literature_payload["literature_report"]["reported_family"] == "hill"
    assert literature_payload["translated_prior"]["sites"]["kd_BMPR1A"]["source"] == "literature_json"


def _write_stub_problem_bundle(tmp_path: Path) -> Path:
    payload = json.loads(PROBLEM_PATH.read_text(encoding="utf-8"))
    payload["data"]["observed_data_ref"] = str(DEFAULT_DATA_PATH)
    payload["metadata"]["local_corpus_dir"] = str(_write_stub_corpus(tmp_path))
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_synthetic_sequential_analysis_run(
    tmp_path: Path,
    prior_mode: str,
    *,
    normalizer_dir: Path | None = None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    normalizer_dir = normalizer_dir or _write_synthetic_normalizer_sources(tmp_path)
    run_dir = tmp_path / f"sequential_{prior_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fit_summary.json").write_text(
        json.dumps(
            {
                "family": "promisys_twostep",
                "prior_mode": prior_mode,
                "cell_lines": ["NMuMG", "BMPR2_KD"],
                "normalizer_data_dir": str(normalizer_dir),
            }
        ),
        encoding="utf-8",
    )
    trace = []
    choices = [("NMuMG", 1), ("BMPR2_KD", 2)]
    bundle = load_bmp4_gradient_data(DEFAULT_DATA_PATH)
    joint = build_joint_bmp4_gradient_data(bundle, cell_lines=["NMuMG", "BMPR2_KD"])
    cell_index = {name: index for index, name in enumerate(joint.cell_lines)}
    for acquisition, (cell_line, dose_index) in enumerate(choices, start=1):
        step_dir = run_dir / f"step_{acquisition:03d}"
        step_dir.mkdir()
        samples = {
            "NMuMG": torch.zeros((4, promisys_twostep.THETA_DIM), dtype=torch.float32),
            "BMPR2_KD": torch.zeros((4, promisys_twostep.THETA_DIM), dtype=torch.float32),
        }
        samples[cell_line] = samples[cell_line] + float(acquisition) * 0.01
        posterior_path = step_dir / "mcmc_posterior_samples.pt"
        torch.save({"theta_norm": samples}, posterior_path)
        idx = cell_index[cell_line]
        dose = float(joint.bmp4_conc[idx, dose_index])
        norm = float(joint.bmp4_conc_norm[idx, dose_index])
        trace.append(
            {
                "acquisition": acquisition,
                "round": acquisition,
                "batch_index": 1,
                "prior_mode": prior_mode,
                "boed": {
                    "best_cell_line": cell_line,
                    "best_bmp4_norm_mu": norm,
                    "best_dose_mu": dose,
                    "best_eig": -0.5 + 0.1 * acquisition,
                    "best_per_cell": [
                        {"cell_line": "NMuMG", "utility": -0.5 + 0.1 * acquisition},
                        {"cell_line": "BMPR2_KD", "utility": -0.6 + 0.1 * acquisition},
                    ],
                },
                "snapped_design": {
                    "cell_line": cell_line,
                    "cell_line_index": idx,
                    "dose_index": dose_index,
                    "dose": dose,
                    "log10_dose": math.log10(max(dose, 1e-30)),
                    "bmp4_norm_design": norm,
                    "proposed_bmp4_norm": norm,
                    "proposed_log10_dose": math.log10(max(dose, 1e-30)),
                    "norm_distance": 0.0,
                    "log10_distance": 0.0,
                    "utility": -0.5 + 0.1 * acquisition,
                },
                "observation": {
                    "x_obs": float(joint.x_obs[idx, dose_index]),
                    "x_obs_norm": float(joint.x_obs_norm[idx, dose_index]),
                },
                "artifacts": {"mcmc_posterior_samples_path": str(posterior_path)},
            }
        )
    (run_dir / "sequential_trace.json").write_text(json.dumps({"trace": trace}), encoding="utf-8")
    return run_dir, normalizer_dir


def _write_stub_corpus(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    body = "\n".join(
        [
            "BMP4 gradient dose-response experiments in NMuMG cells used Bayesian experimental design.",
            "The parameter bottom uses a Normal prior with loc 0.0 and scale 0.5.",
            "The parameter top uses a Normal prior with loc 1.0 and scale 0.5.",
            "The parameter ec50 uses a LogNormal prior with loc 0.0 and scale 0.6.",
            "The parameter hill_n uses a LogNormal prior with loc 0.0 and scale 0.4.",
            "The parameter sigma uses a LogNormal prior with loc -2.0 and scale 0.3.",
            "The parameter kd uses a LogNormal prior with loc -0.2 and scale 0.5.",
            "The parameter weight uses a LogNormal prior with loc 0.0 and scale 0.3.",
            "The parameter s50 uses a LogNormal prior with loc 0.1 and scale 0.6.",
            "The parameter response_hill uses a LogNormal prior with loc 0.0 and scale 0.4.",
            "The parameter sigma_y uses a LogNormal prior with loc -2.0 and scale 0.3.",
            "PyroVI was used for fitting and posterior approximation.",
        ]
    )
    for index in range(3):
        (corpus_dir / f"bmp4_note_{index + 1}.md").write_text(
            f"# BMP4 Literature Note {index + 1}\n\n{body}\n",
            encoding="utf-8",
        )
    return corpus_dir


def _write_synthetic_normalizer_sources(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    source = tmp_path / "normalizer_sources"
    source.mkdir()
    np.save(source / "noised_Ls_4k.npy", rng.gamma(2.0, 100.0, size=(64, 5, 16)).astype("float32"))
    np.save(source / "sim_x_fat_Rs_noised_Ls_4k.npy", rng.gamma(1.5, 0.5, size=(64, 16, 1)).astype("float32"))
    return source


def _stub_literature_responder(prompt: str, tier: str) -> str:
    _ = tier
    if "For each sentence below" in prompt:
        records = []
        for match in re.finditer(r"^\[(\d+)\]\s+(.*)$", prompt, flags=re.MULTILINE):
            sentence_id = int(match.group(1))
            sentence = match.group(2).strip()
            lower = sentence.lower()
            for parameter in sorted(_PRIOR_LIBRARY, key=len, reverse=True):
                if re.search(rf"\b{re.escape(parameter.lower())}\b", lower):
                    spec = _PRIOR_LIBRARY[parameter]
                    records.append(
                        {
                            "id": sentence_id,
                            "type": "prior_distribution",
                            "value": {
                                "parameter": parameter,
                                "distribution": spec["distribution"],
                                "params": spec["params"],
                            },
                        }
                    )
                    break
            if "pyrovi" in lower:
                records.append(
                    {
                        "id": sentence_id,
                        "type": "method_used",
                        "value": {"method": "PyroVI"},
                    }
                )
        return json.dumps(records)

    prior_match = re.search(r"parameter `([^`]+)`", prompt)
    if prior_match:
        parameter = prior_match.group(1)
        spec = _PRIOR_LIBRARY.get(parameter, _PRIOR_LIBRARY["bottom"])
        return json.dumps(
            {
                "distribution": spec["distribution"],
                "params": spec["params"],
                "reasoning": f"stub prior synthesis for {parameter}",
                "cited_papers": ["stub-paper-1", "stub-paper-2", "stub-paper-3"],
            }
        )

    if "rank the candidate BOED backends" in prompt:
        return json.dumps(
            {
                "ranked": ["PyroVI", "MINEBED", "iDAD", "LFIAX"],
                "reasoning": "stub corpus uses PyroVI for variational fitting",
                "cited_papers": ["stub-paper-1", "stub-paper-2", "stub-paper-3"],
            }
        )

    return "{}"
