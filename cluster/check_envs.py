"""Print the versions each conda env actually imports (not what conda/pip lists).

Run with an env's python, e.g.
    ~/miniconda3/envs/tfp_2/bin/python -I cluster/check_envs.py
or for all four envs on this machine:
    for e in num_pyro pymc_env_pm stan3 tfp_2; do ~/miniconda3/envs/$e/bin/python -I cluster/check_envs.py; done
"""
import importlib
import platform
import sys

MODULES = ["numpy", "scipy", "pandas", "jax", "numpyro", "pymc", "pytensor", "arviz",
           "tensorflow", "tensorflow_probability", "tf_keras", "cmdstanpy",
           "matplotlib", "seaborn", "plotly", "tqdm", "modulars"]

env = sys.prefix.rstrip("/").split("/")[-1]
print(f"[{env}] python={platform.python_version()} ({platform.machine()})")
for name in MODULES:
    try:
        mod = importlib.import_module(name)
    except Exception as err:  # not installed in this env, or broken
        print(f"  {name:24s} -  ({type(err).__name__})")
        continue
    version = getattr(mod, "__version__", "")
    if name == "cmdstanpy":
        try:
            version += f"  cmdstan={mod.cmdstan_path()}"
        except Exception:
            version += "  cmdstan=NONE"
    if name == "modulars":
        version = mod.__file__
    print(f"  {name:24s} {version}")
