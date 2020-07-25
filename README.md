# Network_Pharmacology
Network Data Analysis using NetworkX. All the script files to reproduce this study is provided.


EDA_unsupervised.py contains the code used to create plots listed in All_data/node_plots and All_data/edge_score_plots folders. 

mcl_clustering.py contains code to find module in a network.

network_Topology.py contains code to meaure whole netwotk atrributes like average clustering, diameter, shortest path length, transitivity and density. 

network_property.py contains code to measure node scores such as betweenness centrality, closeness centrality, degree, transitivity, eecentricity, and eignvector centrality. Also it contains codes for calculating edge score by using link prediction algorithms, such as jaccard index, preferential attachment score, common neighbor score and resource allocation score. 

# All_data

This study’s primary goal is unraveling the mechanism of action of bioactives of Curcuma longa L. at the molecular level using protein-protein interaction network.
The target proteins (TP) were obtained using similarity ensemble approach and they were further quried in StringDB for retriving the intraction proteins (IP) and a network graph called as true PPIN was created using Networkx library.
Another Network was created by using all the non-existent edges between the list of TP and IP called as false PPIN. 
The PPIN topological measure as edge scores and node scores were calculated and compared between true PPIN and false PPIN. The exploratory data analysis was performed. We identified closeness centrality as important node attribute and jaccard index as important edge attribute for a true PPIN. 

Our All_data folder contains data for all the steps  involved in this project work. The step-wise analysis is explained below with the name of the respective folders containing the data.  

__Step 1: Bioactives_target_proteins__ This folder contains 3 csv files namely Bisdemethoxycurcumin_target_proteins, curcumin_target_proteins and desmethoxycurcumin_target_proteins obtained by searching respective bioactive compounds in (http://sea.bkslab.org/) which gives a list of putative target proteins using similarity ensemble approach. 

__Step 2: Target_proteins+Interacting_proteins__

* A combined list of target proteins (219) were queried in the StringDB protein-protein interaction database for human.
  
* This has led to 208125 interactions for which the interaction score was varied from 150 to 999. Further to reduce the complexity of the network and to increase the confidence of the interaction, we included only edges having interaction score above 300. This has led to total 58482 interactions (edge) involving 11979 proteins (nodes). Out of 11979 proteins, 219 were target proteins (TP) and rest were interacting proteins (TP). These interaction were tabluated in edge.csv in which first column contains TP and second column contains IP. These 58482 interactions were listed in the edge_list_true_PPIN.csv. Also, another network   

__Step 3: Model Building__

* Edward2 ([`tfp.edward2`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/experimental/edward2)):
  A probabilistic programming language for specifying flexible probabilistic
  models as programs. See the
  [Edward2 `README.md`](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/README.md).
* Probabilistic Layers ([`tfp.layers`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers)):
  Neural network layers with uncertainty over the functions they represent,
  extending TensorFlow Layers.

__Layer 3: Probabilistic Inference__

* Markov chain Monte Carlo ([`tfp.mcmc`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc)):
  Algorithms for approximating integrals via sampling. Includes
  [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo),
  random-walk Metropolis-Hastings, and the ability to build custom transition
  kernels.
* Variational Inference ([`tfp.vi`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/vi)):
  Algorithms for approximating integrals via optimization.
* Optimizers ([`tfp.optimizer`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/optimizer)):
  Stochastic optimization methods, extending TensorFlow Optimizers. Includes
  [Stochastic Gradient Langevin Dynamics](http://www.icml-2011.org/papers/398_icmlpaper.pdf).
* Monte Carlo ([`tfp.monte_carlo`](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/monte_carlo)):
  Tools for computing Monte Carlo expectations.

TensorFlow Probability is under active development. Interfaces may change at any
time.

## Examples

See [`tensorflow_probability/examples/`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/)
for end-to-end examples. It includes tutorial notebooks such as:

* [Linear Mixed Effects Models](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Linear_Mixed_Effects_Models.ipynb).
  A hierarchical linear model for sharing statistical strength across examples.
* [Eight Schools](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Eight_Schools.ipynb).
  A hierarchical normal model for exchangeable treatment effects.
* [Hierarchical Linear Models](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/HLM_TFP_R_Stan.ipynb).
  Hierarchical linear models compared among TensorFlow Probability, R, and Stan.
* [Bayesian Gaussian Mixture Models](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Bayesian_Gaussian_Mixture_Model.ipynb).
  Clustering with a probabilistic generative model.
* [Probabilistic Principal Components Analysis](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_PCA.ipynb).
  Dimensionality reduction with latent variables.
* [Gaussian Copulas](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Gaussian_Copula.ipynb).
  Probability distributions for capturing dependence across random variables.
* [TensorFlow Distributions: A Gentle Introduction](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/TensorFlow_Distributions_Tutorial.ipynb).
  Introduction to TensorFlow Distributions.
* [Understanding TensorFlow Distributions Shapes](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb).
  How to distinguish between samples, batches, and events for arbitrarily shaped
  probabilistic computations.
* [TensorFlow Probability Case Study: Covariance Estimation](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/TensorFlow_Probability_Case_Study_Covariance_Estimation.ipynb).
  A user's case study in applying TensorFlow Probability to estimate covariances.

It also includes example scripts such as:

* [Variational Autoencoders](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/vae.py).
  Representation learning with a latent code and variational inference.
* [Vector-Quantized Autoencoder](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/vq_vae.py).
  Discrete representation learning with vector quantization.
* [Disentangled Sequential Variational Autoencoder](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/disentangled_vae.py)
  Disentangled representation learning over sequences with variational inference.
* [Grammar Variational Autoencoder](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/grammar_vae.py).
  Representation learning over productions in a context-free grammar.
* Latent Dirichlet Allocation
  ([Distributions version](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/latent_dirichlet_allocation_distributions.py),
  [Edward2 version](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/latent_dirichlet_allocation_edward2.py)).
  Mixed membership modeling for capturing topics in a document.
+ [Deep Exponential Family](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/deep_exponential_family.py).
  A deep, sparse generative model for discovering a hierarchy of topics.
* [Bayesian Neural Networks](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/bayesian_neural_network.py).
  Neural networks with uncertainty over their weights.
* [Bayesian Logistic Regression](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/logistic_regression.py).
  Bayesian inference for binary classification.

## Installation

### Stable Builds

To install the latest version, run the following:

```shell
# Notes:
# - We recommend that users move towards using TensorFlow 2.x as soon as
#   possible. Until the TF2 stable package is released (due in Sep. 2019),
#   the best way to use TFP with TF2 is to use nightly TFP and TF2 packages:
#     - Nightly TFP: [tfp-nightly](http://pypi.python.org/pypi/tfp-nightly)
#     - Nightly TF2: [tf-nightly-2.0-preview](http://pypi.python.org/pypi/tf-nightly-2.0-preview)
#   Once the TF2 stable release comes out, TFP will issue its 0.8.0 release,
#   which will be tested and stable against TF 2.0.0.
# - You need the latest version of `pip` in order to get the latest version of
#   `tf-nightly-2.0-preview`.
# - For GPU TF, use `tf-nightly-2.0-preview-gpu`.
# - The `--upgrade` flag ensures you'll get the latest version.
# - The `--user` flag ensures the packages are installed to your user directory
#   rather than the system directory.
python -m pip install pip --upgrade --user
python -m pip install tf-nightly-2.0-preview tfp-nightly --upgrade --user
TFVERSION=$(python -c 'import tensorflow; print(tensorflow.__version__)')
# If you have an older pip, you might get this older version of
# tf-nightly-2.0-preview, so check to be sure.
[[ $TFVERSION == '2.0.0-dev20190731' ]] &&
  echo >&2 "Failed to install the most recent TF. Found: ${TFVERSION}."
```

TensorFlow Probability depends on a recent stable release of
[TensorFlow](https://www.tensorflow.org/install) (pip package `tensorflow`). See
the [TFP release notes](https://github.com/tensorflow/probability/releases) for
details about dependencies between TensorFlow and TensorFlow Probability.

Note: Since TensorFlow is *not* included as a dependency of the TensorFlow
Probability package (in `setup.py`), you must explicitly install the TensorFlow
package (`tensorflow` or `tensorflow-gpu`). This allows us to maintain one
package instead of separate packages for CPU and GPU-enabled TensorFlow.

To force a Python 3-specific install, replace `pip` with `pip3` in the above
commands. For additional installation help, guidance installing prerequisites,
and (optionally) setting up virtual environments, see the [TensorFlow
installation guide](https://www.tensorflow.org/install).

### Nightly Builds

There are also nightly builds of TensorFlow Probability under the pip package
`tfp-nightly`, which depends on one of `tf-nightly`, `tf-nightly-gpu`,
`tf-nightly-2.0-preview` or `tf-nightly-gpu-2.0-preview`. Nightly builds include
newer features, but may be less stable than the versioned releases. Docs are
periodically refreshed [here](
https://github.com/tensorflow/probability/blob/master/tensorflow_probability/g3doc/api_docs/python/tfp.md).

### Installing from Source

You can also install from source. This requires the [Bazel](
https://bazel.build/) build system.

```shell
# sudo apt-get install bazel git python-pip  # Ubuntu; others, see above links.
git clone https://github.com/tensorflow/probability.git
cd probability
bazel build --copt=-O3 --copt=-march=native :pip_pkg
PKGDIR=$(mktemp -d)
./bazel-bin/pip_pkg $PKGDIR
pip install --user --upgrade $PKGDIR/*.whl
```

## Community

As part of TensorFlow, we're committed to fostering an open and welcoming
environment.

* [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow): Ask
  or answer technical questions.
* [GitHub](https://github.com/tensorflow/probability/issues): Report bugs or
  make feature requests.
* [TensorFlow Blog](https://medium.com/tensorflow): Stay up to date on content
  from the TensorFlow team and best articles from the community.
* [Youtube Channel](http://youtube.com/tensorflow/): Follow TensorFlow shows.
* [tfprobability@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/tfprobability):
  Open mailing list for discussion and questions.

See the [TensorFlow Community](https://www.tensorflow.org/community/) page for
more details. Check out our latest publicity here:

+ [Coffee with a Googler: Probabilistic Machine Learning in TensorFlow](
  https://www.youtube.com/watch?v=BjUkL8DFH5Q)
+ [Introducing TensorFlow Probability](
  https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245)

## Contributing

We're eager to collaborate with you! See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for a guide on how to contribute. This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

## References

If you use TensorFlow Probability in a paper, please cite: 

+ _TensorFlow Distributions._ Joshua V. Dillon, Ian Langmore, Dustin Tran,
Eugene Brevdo, Srinivas Vasudevan, Dave Moore, Brian Patton, Alex Alemi, Matt
Hoffman, Rif A. Saurous.
[arXiv preprint arXiv:1711.10604, 2017](https://arxiv.org/abs/1711.10604).

(We're aware there's a lot more to TensorFlow Probability than Distributions, but the Distributions paper lays out our vision and is a fine thing to cite for now.)
