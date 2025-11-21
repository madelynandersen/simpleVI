# print_versions.py
def try_import(name, attr="__version__", alias=None):
    try:
        mod = __import__(name) if alias is None else __import__(alias)
        # If we imported an alias (e.g., tensorflow_probability as tfp), re-get the object
        if alias is not None:
            mod = globals()[alias]
        ver = getattr(mod, attr, None)
        print(f"{name:<28} {ver}")
    except Exception as e:
        print(f"{name:<28} (not installed)")

# Core Python + numerics
try_import("python", attr="version")
try:
    import sys
    print(f"{'python (sys)':<28} {sys.version.split()[0]}")
except:
    pass
try_import("numpy")
try_import("scipy")

# PyTorch / JAX / TF stacks
try_import("torch")
try_import("jax")
try_import("jaxlib")
try_import("tensorflow")
try:
    import tensorflow_probability as tfp  # noqa
    print(f"{'tensorflow_probability':<28} {tfp.__version__}")
except Exception:
    print(f"{'tensorflow_probability':<28} (not installed)")

# PPLs
try_import("pymc")
try_import("arviz")
try_import("numpyro")
try_import("pyro")

# Stan interfaces (optional)
try_import("cmdstanpy")
try_import("pystan")

# For completeness: matplotlib for plotting versions if needed
try_import("matplotlib")
