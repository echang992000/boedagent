#!/usr/bin/env python3
"""Post-install smoke test for boed-agent.

Run after ``pip install -e .`` (or any of the optional extras) to
verify the install is healthy end-to-end.  Exits 0 on success, 1 on
any failure.  Intended to be CI-friendly: no network calls, no GPU,
no external binaries required for the default run.

Typical usage
-------------

    # From the repo root, in the environment you just installed into:
    python scripts/smoketest.py

    # Strict mode — also runs the dry-run literature example
    # (still offline, uses LocalCorpusClient + RecordingLLMClient):
    python scripts/smoketest.py --with-example

    # Include the pytest suite:
    python scripts/smoketest.py --with-tests

    # Full run:
    python scripts/smoketest.py --with-example --with-tests

What it checks
--------------

1. Python version satisfies the package's ``requires-python``.
2. Core package imports (``boed_agent``, ``BOEDAgent``, protocols).
3. Optional extras import when installed (``pyro``, ``lfiax``,
   ``hdbscan``, ``pypdf``, ``tenacity``, ``openai``, ``anthropic``).
4. ``boed-agent`` CLI entry point resolves and ``list-backends`` runs.
5. Torch device helper reports a sensible device (cpu / cuda / mps).
6. (Optional) Runs ``examples/agent/lit_only_dry_run.py`` end-to-end.
7. (Optional) Runs the offline pytest suite.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Check harness
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    status: str  # "pass" | "skip" | "fail"
    detail: str = ""


class Reporter:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def run(
        self,
        name: str,
        fn: Callable[[], str | None],
        *,
        required: bool = True,
    ) -> None:
        """Run ``fn``; interpret ``None`` / a string as success detail,
        ``SkipCheck`` as a skip, any other exception as failure.

        Required checks fail the overall run; non-required checks only
        warn so optional extras don't break the smoke test for users
        who haven't installed them.
        """
        try:
            detail = fn() or ""
            self.checks.append(Check(name=name, status="pass", detail=detail))
            print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
        except SkipCheck as exc:
            self.checks.append(Check(name=name, status="skip", detail=str(exc)))
            print(f"  [SKIP] {name} — {exc}")
        except Exception as exc:
            status = "fail" if required else "skip"
            self.checks.append(Check(name=name, status=status, detail=str(exc)))
            tag = "[FAIL]" if required else "[WARN]"
            print(f"  {tag} {name} — {exc}")

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    def summary(self) -> str:
        passed = sum(c.status == "pass" for c in self.checks)
        skipped = sum(c.status == "skip" for c in self.checks)
        failed = sum(c.status == "fail" for c in self.checks)
        return f"{passed} passed, {skipped} skipped, {failed} failed"


class SkipCheck(Exception):
    """Raised inside a check to indicate the prerequisite is absent."""


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version() -> str:
    major, minor = sys.version_info.major, sys.version_info.minor
    if (major, minor) < (3, 11):
        raise RuntimeError(
            f"Python {major}.{minor} detected; boed-agent requires >=3.11"
        )
    return f"Python {major}.{minor}.{sys.version_info.micro}"


def check_core_imports() -> str:
    from boed_agent import (  # noqa: F401
        BOEDAgent,
        ParameterInfo,
        SimpleSimulator,
        SimulatorMetadata,
        TokenBudget,
    )
    from boed_agent.simulator_protocol import Simulator, introspect_metadata  # noqa: F401
    from boed_agent.literature.trace import (  # noqa: F401
        LiveCitationError,
        ReasoningStep,
        ReasoningTrace,
        step_is_grounded,
        validate_citations,
    )
    from boed_agent.literature.clients import (  # noqa: F401
        ArxivClient,
        OpenAlexClient,
        PubMedClient,
        SemanticScholarClient,
        UnpaywallClient,
    )
    from boed_agent.literature.clients.base import with_retries  # noqa: F401
    return "all core symbols imported"


def check_optional_import(module_name: str, *, extra: str) -> Callable[[], str]:
    def runner() -> str:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise SkipCheck(f"{module_name} not installed (pip install -e \".[{extra}]\") — {exc}")
        version = getattr(module, "__version__", "unknown")
        return f"{module_name} {version}"

    return runner


def check_device_helper() -> str:
    try:
        from boed_agent.utils.device import device_summary
    except ImportError as exc:
        raise RuntimeError(f"utils.device not importable: {exc}")
    summary = device_summary()
    return (
        f"torch={summary.get('torch')} mps={summary['mps']} cuda={summary['cuda']} "
        f"chosen={summary['chosen']}"
    )


def check_cli_entrypoint() -> str:
    cli = shutil.which("boed-agent")
    if cli is None:
        raise RuntimeError(
            "`boed-agent` entry point not on PATH — did you run `pip install -e .`?"
        )
    try:
        out = subprocess.run(
            [cli, "list-backends"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"`boed-agent list-backends` failed: {exc.stderr.strip()}")
    first_line = out.stdout.strip().splitlines()[:1]
    return f"{cli} → {first_line[0] if first_line else '(empty output)'}"


def check_dispatcher_waterfall() -> str:
    """Exercise every branch of SimulatorChoiceModule against in-memory
    simulators.  Catches any regression in the protocol dispatch without
    touching the heavier backends."""
    from boed_agent import SimpleSimulator, SimulatorMetadata
    from boed_agent.backends.registry import BackendRegistry
    from boed_agent.simulator_choice import SimulatorChoiceModule

    registry = BackendRegistry.default()
    cases = [
        (True, False, "pyro"),
        (False, True, ("idad", "minebed")),
        (False, False, "lfiax"),
    ]
    picks: list[str] = []
    for is_explicit, is_diff, expected in cases:
        sim = SimpleSimulator(
            fn=lambda theta, xi: theta,
            metadata=SimulatorMetadata(),
            is_explicit=is_explicit,
            is_differentiable=is_diff,
        )
        choice = SimulatorChoiceModule.select(
            sim, registry=registry, backend_options={}
        )
        picks.append(choice.backend.name)
        if isinstance(expected, tuple):
            if choice.backend.name not in expected:
                raise RuntimeError(
                    f"({is_explicit=}, {is_diff=}) picked {choice.backend.name}, expected one of {expected}"
                )
        else:
            if choice.backend.name != expected:
                raise RuntimeError(
                    f"({is_explicit=}, {is_diff=}) picked {choice.backend.name}, expected {expected}"
                )
    return " → ".join(picks)


def check_dry_run_example(repo_root: Path) -> str:
    example = repo_root / "examples" / "agent" / "lit_only_dry_run.py"
    if not example.exists():
        raise RuntimeError(f"expected example not found: {example}")
    out = subprocess.run(
        [sys.executable, str(example)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    stdout = out.stdout
    if "Backend:" not in stdout or "Literature Reasoning Trace" not in stdout:
        raise RuntimeError("dry-run example output missing expected sections")
    return f"{len(stdout.splitlines())} lines of reasoning-trace markdown produced"


def check_pytest_suite(repo_root: Path) -> str:
    import pytest  # noqa: F401

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "tests/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    tail = "\n".join(result.stdout.strip().splitlines()[-3:])
    if result.returncode != 0:
        raise RuntimeError(f"pytest failed:\n{tail}\nstderr:\n{result.stderr[-500:]}")
    return tail.replace("\n", " | ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--with-example",
        action="store_true",
        help="run the offline dry-run literature example",
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="run the pytest suite (offline tests only; excludes -m live)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    print("boed-agent install smoke test")
    print("=" * 60)
    print(f"Python executable : {sys.executable}")
    print(f"Repo root         : {repo_root}")
    print()

    r = Reporter()

    # Required checks — any failure here means the install is broken.
    print("Required checks")
    r.run("python >= 3.11", check_python_version)
    r.run("core imports", check_core_imports)
    r.run("CLI `boed-agent list-backends`", check_cli_entrypoint)
    r.run("dispatcher waterfall", check_dispatcher_waterfall)

    # Informational checks — these report extras state without failing.
    print()
    print("Optional extras (install with `pip install -e \".[<extra>]\"`)")
    r.run("pyro (for [pyro])", check_optional_import("pyro", extra="pyro"), required=False)
    r.run("torch", check_optional_import("torch", extra="pyro"), required=False)
    r.run("openai (for [agents])", check_optional_import("openai", extra="agents"), required=False)
    r.run("anthropic (for [agents])", check_optional_import("anthropic", extra="agents"), required=False)
    r.run("pypdf (for [literature])", check_optional_import("pypdf", extra="literature"), required=False)
    r.run("tenacity (for [literature])", check_optional_import("tenacity", extra="literature"), required=False)
    r.run("hdbscan (for [literature])", check_optional_import("hdbscan", extra="literature"), required=False)
    r.run("scikit-learn (for [literature])", check_optional_import("sklearn", extra="literature"), required=False)
    r.run("lfiax (for [lfiax])", check_optional_import("lfiax", extra="lfiax"), required=False)
    r.run("PyYAML (for [yaml])", check_optional_import("yaml", extra="yaml"), required=False)
    r.run("matplotlib (for [plot])", check_optional_import("matplotlib", extra="plot"), required=False)

    # Device hint — always run; only warns on failure.
    print()
    print("Device detection")
    r.run("torch device summary", check_device_helper, required=False)

    # Optional heavier checks — opt-in via flags so the default run stays fast.
    if args.with_example:
        print()
        print("End-to-end dry run")
        r.run("examples/agent/lit_only_dry_run.py", lambda: check_dry_run_example(repo_root))

    if args.with_tests:
        print()
        print("Test suite")
        r.run("pytest tests/", lambda: check_pytest_suite(repo_root))

    print()
    print("=" * 60)
    print(f"Summary: {r.summary()}")
    if r.failed:
        print()
        print(textwrap.dedent("""
            One or more required checks failed.  Common causes on a fresh install:
              - wrong Python version (need >= 3.11)
              - forgot `pip install -e .` in the active environment
              - shell PATH does not include the venv/conda `bin/` dir

            Re-run inside the correct environment.  To just re-check imports:
              python scripts/smoketest.py
        """).strip())
        return 1

    print("All required checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
