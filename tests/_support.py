"""Shared helpers for the test suite.

Each package lives in its own conda environment, so tests that need a package are skipped
when it isn't installed. Run the suite once per environment:

    python -m unittest discover -s tests -t .
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNERS_DIR = REPO_ROOT / "paper_vignettes" / "overall_bias_runners"
SINGLE_MC = REPO_ROOT / "single_MC"
for path in (REPO_ROOT, RUNNERS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def has_module(name):
    return importlib.util.find_spec(name) is not None


def requires(*modules):
    missing = [m for m in modules if not has_module(m)]
    return unittest.skipIf(bool(missing), f"{', '.join(missing)} not installed in this environment")


ENV_CMDSTAN = Path(sys.prefix) / "bin" / "cmdstan"


def requires_cmdstan():
    """Stan tests run where the paper's CmdStan is set up: the stan3 env's bundled CmdStan, or SIMPLEVI_CMDSTAN_PATH."""
    available = has_module("cmdstanpy") and (bool(os.environ.get("SIMPLEVI_CMDSTAN_PATH")) or ENV_CMDSTAN.exists())
    return unittest.skipUnless(available, "no CmdStan for this environment (run the Stan tests in stan3)")


def set_cmdstan_path():
    from cmdstanpy import set_cmdstan_path as _set
    _set(os.environ.get("SIMPLEVI_CMDSTAN_PATH") or str(ENV_CMDSTAN))
