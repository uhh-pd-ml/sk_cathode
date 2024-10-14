# sk_cathode

**Table of Contents**
- [sk\_cathode](#sk_cathode)
  - [Core idea](#core-idea)
  - [Demos](#demos)
  - [Installation](#installation)
    - [Docker image](#docker-image)
    - [Installation via conda/pip](#installation-via-condapip)
  - [Shared class methods](#shared-class-methods)
    - [Generative model methods](#generative-model-methods)
    - [Classifier methods](#classifier-methods)
    - [Preprocessing scaler methods](#preprocessing-scaler-methods)
  - [Overview of existing building blocks](#overview-of-existing-building-blocks)
    - [Generative models](#generative-models)
    - [Classifier models](#classifier-models)
    - [Preprocessing and pipelines](#preprocessing-and-pipelines)
    - [Ensembling](#ensembling)
    - [Evaluation](#evaluation)
  - [Contributing](#contributing)


## Core idea

This (work-in-progress) repo aims to illustrate how to deploy anomaly detection models such as [CATHODE](https://arxiv.org/abs/2109.00546) and [LaCATHODE](https://arxiv.org/abs/2210.14924) by hiding technical implementation details behind a scikit-learn-like API.

The directory `sk_cathode/` provides simple-to-use classes and functions, to stick together just like Lego pieces. The directory `demos/` provides illustratives jupyter notebooks on how these can be brought together. They are briefly summarized below.

The primary goal is to make these anomaly detection methods more accessible and easy to play with, within the scope of tutorials and proof-of-concept studies.

## Demos

- `demos/weak_supervision.ipynb` gives a brief overview of the core idea of weak supervision for anomaly detection, using a particle physics dataset. It defines the so-called Idealized Anomaly Detector and compares it to a fully supervised approach. It also features some slim code to compare methods in terms of median performance with 68% confidence intervals.
- `demos/weak_supervision_gauss_example.ipynb` illustrates the core idea of weak supervision with a Gaussian toy dataset and shows at which point it performs similarly to a fully supervised classifier.
- `demos/cathode_walkthrough.ipynb` simply describes the basic working principle of the CATHODE method for weakly supervised anomaly detection on the LHCO example dataset.
- `demos/lacathode_walkthrough.ipynb` touches the issue of background sculpting and guides through the working principle of LaCATHODE to mitigate this in (CATHODE-based) weak supervision.
- `demos/anode_walkthrough.ipynb` gives a brief overview of the ANODE method (CATHODE's predecessor) for anomaly detection, which works analogous to weak supervision but constructs an explicit likelihood ratio using normalizing flows instead of training a classifier.
- `demos/tree_classifier.ipynb` discusses the challenge of uninformative features in weakly supervised anomaly detection and demonstrates substantial improvement by using ensembles of boosted decision trees instead of neural networks.
- `demos/autoencoder_gauss.ipynb` introduces a different paradigm for anomaly detection, namely outlier detection based on autoencoders. The working principle is illustrated on a Gaussian toy dataset.

## Installation

### Docker image

We provide a Docker image with all necessary dependencies installed.
To start an interactive container **with Docker**, one can use the following
command:

```bash
docker run -it --rm jobirk/sk_cathode:latest bash
```

To start an interactive container **with Singularity**, one can use the
following command:
```bash
singularity shell docker://jobirk/sk_cathode:latest
```

General info about how to run a singularity container in combination with VSCode, can be 
found in the [instructions in `joschkabirk/docker-template`](https://github.com/joschkabirk/docker-template?tab=readme-ov-file#set-up-vscode-for-remote-development-with-singularity).

The instructions below are tested on the DESY Maxwell cluster.

1. Create folders for the Singularity cache and temporary files:
```bash
mkdir -p /gpfs/dust/maxwell/user/$USER/.singularity/cache
mkdir -p /gpfs/dust/maxwell/user/$USER/.singularity/tmp
mkdir -p /gpfs/dust/maxwell/user/$USER/singularity_images
```
2. Tell singularity to use these folders by adding the following lines to your `~/.bashrc`/`~/.zshrc`:
```bash
export SINGULARITY_CACHEDIR=/gpfs/dust/maxwell/user/$USER/.singularity/cache
export SINGULARITY_TMPDIR=/gpfs/dust/maxwell/user/$USER/.singularity/tmp
```
3. Pull the image from the Docker Hub and convert into a Singularity image.
NOTE: this takes quite some time and is not required if a colleague (who works
on the same cluster) has already done this and provides you access to their
`.sif` file.
```bash
singularity build /gpfs/dust/maxwell/user/$USER/singularity_images/sk_cathode.sif docker://jobirk/sk_cathode:latest
```
4. Setup the VSCode connection to Maxwell. Add the following entry to your `~/.ssh/config`. This tells VSCode to start the whole remote development environment with Singularity on Maxwell.
```
Host sk_cathode_demo
    RemoteCommand export SINGULARITY_CACHEDIR=<your_cachedir> && export SINGULARITY_TMPDIR=<your_tmpdir> && singularity exec --nv -B /beegfs/desy/user -B /gpfs/dust/maxwell/user <path_to_the_sif_file> /bin/zsh
    User <your_username>
    HostName max-display004.desy.de
    RequestTTY yes
```
5. Set the install path of the corresponding VSCode container extension in the `settings.json` file (local VSCode settings):
```json
"remote.SSH.serverInstallPath": {
    "sk_cathode_demo": "<some_path_where_you_have_enough_space>"
}
```
6. Now connect to this host in VSCode, open your folder where you have the things you want
to run, and install the jupyter extension (to run notebooks in VSCode):
![](https://syncandshare.desy.de/index.php/s/m355983ZtRxFYzx/download)


### Installation via conda/pip

Just clone via the usual way, for example:

```bash
git clone git@github.com:uhh-pd-ml/sk_cathode.git
cd sk_cathode
```

The `requirements.txt` covers the necessary libraries to run the default demo notebook. For example, one can create a fresh conda environment and install the dependencies with:

```bash
conda create -n "sk_cathode_env" python=3.9.7
conda activate sk_cathode_env
pip install -r requirements.txt
```

In order to run the Normalizing Flow with Pyro backend, one further needs to install [Pyro](https://pyro.ai/). For the Conditional Flow Matching generative model, one needs to install [torchdyn](https://torchdyn.org/). The `requirements_full.txt` includes these additional dependencies.

## Shared class methods

Just as scikit-learn estimators, the building blocks of `sk_cathode` share common methods, so they are easily interchangeable and can be stacked in a pipeline. A key extension to scikit-learn is that the jacobian determinants are tracked through the preprocessing steps, which is crucial to correctly transform estimated densities (if the dedicated extended pipeline class is used). Moreover, the extended pipeline supports conditional sampling.

The shared methods per type of estimator are summarized below. In some models, not all methods might be implemented, either because of model-specific limitations or because no one had the time to do it (yet).

In addition to an extended pipeline class, there is an ensemble model wrapper, which stacks multiple models in parallel and thus supports most methods below.

### Generative model methods

- `fit(X, m, X_val=None, m_val=None)`: Trains the model to learn `X` conditioned on `m`.
- `transform(X, m=None)`: Transforms the provided data to the latent space.
- `fit_transform(X, m, X_val=None, m_val=None)`: Trains the model and transforms the provided data to the latent space.
- `inverse_transform(Xt, m=None)`: Transforms the latent space data to the original space.
- `[log_]jacobian_determinant(X, m=None)`: Computes the [log] jacobian determinant of the transformation.
- `inverse_[log_]jacobian_determinant(Xt, m=None)`: Computes the [log] jacobian determinant of the inverse transformation.
- `sample(n_samples=1, m=None)`: Samples from the model with conditional values `m`.
- `predict_[log_]proba(X, m=None)`: Computes the [log] likelihood of the data (per data point).
- `score_samples(X, m=None)`: Alias for `predict_log_proba`.
- `score(X, m=None)`: Computes the (summed) log likelihood over the full dataset.
- `load_best_model()`: Loads the best model state from the provided save_path.
- `load_train_loss()`: Loads the training loss from the provided save_path.
- `load_val_loss()`: Loads the validation loss from the provided save_path.
- `load_epoch_model(epoch)`: Loads the model state from the provided save_path at a specific epoch.

### Classifier methods

- `fit(X, y, X_val=None, y_val=None, sample_weight=None, sample_weight_val=None)`: Trains the model with data `X` and labels `y`.
- `predict(X)`: Predicts the labels for the provided data.
- `load_best_model()`: Loads the best model state from the provided save_path.
- `load_train_loss()`: Loads the training loss from the provided save_path.
- `load_val_loss()`: Loads the validation loss from the provided save_path.
- `load_epoch_model(epoch)`: Loads the model state from the provided save_path at a specific epoch.

### Preprocessing scaler methods

- `fit(X, y=None)`: Fits the preprocessing step to the provided data.
- `transform(X)`: Transforms the provided data.
- `fit_transform(X, y=None)`: Fits the preprocessing step to the provided data and transforms it.
- `inverse_transform(X)`: Inverse-transforms the provided data.
- `[log_]jacobian_determinant(X)`: Computes the [log] jacobian determinant of the transformation (important for density estimation with preprocessing).
- `inverse_[log_]jacobian_determinant(X)`: Computes the [log] jacobian determinant of the inverse transformation.

## Overview of existing building blocks

Below is a brief overview of the building blocks provided in `sk_cathode`, out of which one can build anomaly detection pipelines.

### Generative models

- `sk_cathode.generative_models.conditional_normalizing_flow_torch.ConditionalNormalizingFlow`: A conditional normalizing flow model  implemented directly in PyTorch.
- `sk_cathode.generative_models.conditional_normalizing_flow_pyro.ConditionalNormalizingFlow`: A conditional normalizing flow model  implemented via the Pyro library.
- `sk_cathode.generative_models.conditional_normalizing_flow_nflows.ConditionalNormalizingFlow`: A conditional normalizing flow model  implemented via nFlows.
- `sk_cathode.generative_models.conditional_flow_matching.ConditionalFlowMatching`: A conditional flow matching model  implemented via PyTorch and the torchdyn ODE solver.
- `sk_cathode.generative_models.autoencoder.Autoencoder`: A simple autoencoder based on a fully connected neural network, implemented in PyTorch.

### Classifier models

- `sk_cathode.classifier_models.neural_network_classifier.NeuralNetworkClassifier`: A simple neural network classifier, implemented in PyTorch.
- `sk_cathode.classifier_models.boosted_decision_tree.HGBClassifier`: A BDT classifier based on the HistGradientBoosting algorithm, implemented in scikit-learn.

### Preprocessing and pipelines

- `sk_cathode.utils.preprocessing.LogitScaler`: Preprocessing scaler class that performs a logit transformation.
- `sk_cathode.utils.preprocessing.ExtStandardScaler`: Preprocessing scaler class that extends the scikit-learn StandardScaler to track jacobian determinants for correct density estimation.
- `sk_cathode.utils.preprocessing.ExtPipeline`: Extension of scikit-learn pipeline to track jacobian determinants for correct density estimation.
- `sk_cathode.utils.preprocessing.make_ext_pipeline`: Function for constructing an `ExtPipeline` object on a list of steps.

### Ensembling

- `sk_cathode.utils.ensembling_utils.EnsembleModel`: Wrapper class to combine multiple models into a single one.
- `sk_cathode.utils.ensembling_utils.EpochEnsembleModel`: Wrapper class to build an ensemble from multiple epochs of the same training.

### Evaluation

- `sk_cathode.utils.evaluation_functions.sic_curve`: Function to compute the SIC curve.
- `sk_cathode.utils.evaluation_functions.hists_2d`: Function to plot 2D correlation histograms.
- `sk_cathode.utils.evaluation_functions.pulls_2d`: Function to plot 2D pulls between two multidimensional distributions in pairs.
- `sk_cathode.utils.evaluation_functions.preds_2d`: Function to plot model predictions as a function of the input features pair-wise in 2D.
- `sk_cathode.utils.evaluation_functions.plot_roc_and_sic`: Quick function to plot both the ROC and the SIC curve.

## Contributing

The repo is very much open for contributions via pull requests, e.g. to add more classifiers and generative models, other ways of doing anomaly detection, or just an illustrative demo notebook highlighting a particular study. For questions, feel free to contact `manuel.sommerhalder[at]uni-hamburg.de`.
