# simpleVI

"The Curious Case of the Default Settings: Evaluating Default Performance of Variational Inference
Software" (citation incoming)

For reproduction of our figures and results described in our posted manuscript, go to [Examples and Vignettes](#examples-and-vignettes).

simpleVI is a modular framework for experimenting with variational inference methods across different probabilistic programming backends. Currently, examples exist for PyMC, NumPyro, and TensorFlow Probability. Future updates will include Stan.

This framework is under active development, including ongoing Stan VI implementation and experimentation. Any Stan VI files within this framework and experiments not described in the manuscript are not currently guaranteed to be stable.

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

Notebooks in `paper_vignettes` use some of these modular components to generate the plots for our full manuscript, but, in order to make it simplest for users to work with PPLs themselves, do not modularize the specific usage of the PPLs.

The `single_MC` folder contains subfolders for each specific experiment we performed. In each subfolder, for example, `single_MC/1dgaussian_knownvar`, the `random_restarts` folder contains the code for performing our random restart analysis in each PPL. The files in `single_MC/1dgaussian_knownvar` contain code to perform a single-restart analysis for each PPL.


## How to use this framework

A recommended place to start is skimming through the notebooks in `paper_vignettes` to see examples used in the full manuscript.

To learn how to use our modular components, we recommend selecting an experiment in `single_MC` and reviewing the random-restart analysis code.


## Examples and Vignettes

For examples and vignettes, including those demonstrated in our manuscript, you may consider the following notebooks:
- [An example analysis notebook](paper_vignettes/example_analysis.ipynb)
-[Manuscript section 3.1: Figure showing the uncertainty bias produced by the default settings in PyMC, shown in a 1d gaussian with known variance](single_MC/1dgaussian_knownvar
/3_1_pymc.ipynb) (Note that this notebook is 300k iterations, while the manuscript demonstrates 3 million iterations. The manuscript's figure may be recreated by changing the number of iterations.)
- [Manuscript section 3.2: Notebook demonstrating how initialization works in NumPyro](paper_vignettes/3_2_numpyro_initialization.ipynb)
- [Manuscript section 3.2: Notebook demonstrating how initialization works in TFP](paper_vignettes/3_2_tfp_initialization.ipynb)
- [Manuscript section 3.2: Notebook demonstrating how initialization works in PyMC](paper_vignettes/3_2_pymc_initialization.ipynb)
- [Manuscript section 3.2: Notebook demonstrating how support points work in PyMC](paper_vignettes/3_2_pymc_support_points.ipynb)
- [Manuscript section 3.3: Notebook and figures demonstrating the default behavior of TFP and NumPyro and how the behavior of TFP changes when we add transformations](paper_vignettes/3_3_tfp_transforms_matter.ipynb)
- [Manuscript section 3.5: Notebook and figures demonstrating initialization sensitivity](paper_vignettes/3_5_numpyro_betamix_initialization_challenges.ipynb)

For an example of a random restart analysis, we encourage you to look at the [PyMC restart analysis of a 1d conjugate Gaussian with known variance](single_MC/1dgaussian_knownvar/random_restarts/rr_pymc.ipynb), the [NumPyro restart analysis of a 1d conjugate Gaussian with known variance](single_MC/1dgaussian_knownvar/random_restarts/rr_numpyro.ipynb), the [TFP restart analysis of a 1d conjugate Gaussian with known variance](single_MC/1dgaussian_knownvar/random_restarts/rr_tfp.ipynb), the [Pyro restart analysis of a 1d conjugate Gaussian with known variance](single_MC/1dgaussian_knownvar/random_restarts/rr_pyro.ipynb), or the [Stan restart analysis of a 1d conjugate Gaussian with known variance.](single_MC/1dgaussian_knownvar/random_restarts/rr_stan.ipynb)

Other experiments we ran may be found in the [experiments folder single_MC](single_MC)

