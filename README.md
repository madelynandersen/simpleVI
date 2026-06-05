# simpleVI

simpleVI is a modular framework for experimenting with variational inference methods across different probabilistic programming backends. Currently examples exist for PyMC, NumPyro, and TensorFlow Probability. Future updates will include Stan.

This framework is under active development on the following: ongoing Stan VI implementation and experimentation. Any Stan VI files within this framework are currently not guaranteed to be stable.

## Installation

From the project root (where this README and `pyproject.toml` live):

pip install -e .

For each conda environment, use:   
 pip install -e ".[numpyro]"
    pip install -e ".[pymc]"
    pip install -e ".[tfp]"
    pip install -e ".[all]"

Note that the script at the top of any PyMC notebooks may need to be changed to represent your specific PyMC installation. On our machine, we use `import pytensor; pytensor.config.cxx = '/usr/bin/clang++'` but this may differ by system.

## What is in this framework/repository?

The **core reusable code** lives in the `modulars` package. Reusable code includes plotting, transformations, running random restarts of PPLs, automating PPL implementation, and other functionality to support our work.

Notebooks in `paper_vignettes` use some of these modular components to generate the plots for our full manuscript but, in order to make it simplest for users to work with PPLs themselves, do not modularize the specific usage of the PPLs.

The `single_MC` folder contains subfolders for each specific experiment we performed. In each subfolder, for example `single_MC/1dgaussian_knownvar`, the `random_restarts` folder contains the code for performing our random restart analysis in each PPL. The files in `single_MC/1dgaussian_knownvar` contain code for performing a single restart analysis in each PPL.


## How to use this framework

A recommended place to start is skimming through the notebooks in `paper_vignettes` to see examples used in the full manuscript.

To learn how to use our modular components, we recommend choosing an experiment in `single_MC` and reading the random restart analysis code.




