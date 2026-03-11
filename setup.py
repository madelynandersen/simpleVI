from setuptools import setup, find_packages

setup(
    name="simplevi",
    version="0.1.0",
    description="A simple, modular framework for variational inference experiments.",
    author="Madelyn Andersen",
    author_email="madelyna@mit.edu",
    license="MIT",
    packages=find_packages(include=["modulars", "modulars.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22",
        "matplotlib>=3.7",
    ],
    extras_require={
        "numpyro": [
            "jax>=0.4",
            "jaxlib>=0.4",
            "numpyro>=0.13",
        ],
        "pymc": [
            "pymc>=5.0",
            "pytensor>=2.0",
        ],
        "tfp": [
            "tensorflow>=2.15",
            "tensorflow_probability>=0.24",
        ],
        "all": [
            "jax>=0.4",
            "jaxlib>=0.4",
            "numpyro>=0.13",
            "pymc>=5.0",
            "pytensor>=2.0",
            "tensorflow>=2.15",
            "tensorflow_probability>=0.24",
        ],
    },
)