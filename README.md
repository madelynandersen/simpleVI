# simpleVI

simpleVI is a small, modular framework for experimenting with variational inference methods across different probabilistic programming backends.

The **core reusable code** lives in the `modulars` package. Notebooks and scripts in `single_MC` and elsewhere use these modular components to run specific experiments.

## Installation

From the project root (where this README and `pyproject.toml` live):

pip install -e .

For each conda environment, use:   
 pip install -e ".[numpyro]"
    pip install -e ".[pymc]"
    pip install -e ".[tfp]"
    pip install -e ".[all]"
