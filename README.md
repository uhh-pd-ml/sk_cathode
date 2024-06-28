# sk_cathode

## Core idea

This (work-in-progress) repo aims to illustrate how to deploy anomaly detection models such as [CATHODE](https://arxiv.org/abs/2109.00546) and [LaCATHODE](https://arxiv.org/abs/2210.14924) by hiding technical implementation details behind a scikit-learn-like API.

The directory `sk_cathode/` provides simple-to-use classes and functions, to stick together just like Lego pieces. The directory `demos/` provides illustratives jupyter notebooks on how these can be brought together. They are briefly summarized below.

The primary goal is to make these anomaly detection methods more accessible and easy to play with, within the scope of tutorials and proof-of-concept studies.

## Demos

- `demos/weak_supervision.ipynb` gives a brief overview of the core idea of weak supervision for anomaly detection. It defines the so-called Idealized Anomaly Detector and compares it to a fully supervised approach. It also features some slim code to compare methods in terms of median performance with 68% confidence intervals.
- `demos/cathode_walkthrough.ipynb` simply describes the basic working principle of the CATHODE method for weakly supervised anomaly detection on the LHCO example dataset.
- `demos/lacathode_walkthrough.ipynb` touches the issue of background sculpting and guides through the working principle of LaCATHODE to mitigate this in (CATHODE-based) weak supervision.
- `demos/anode_walkthrough.ipynb` gives a brief overview of the ANODE method (CATHODE's predecessor) for anomaly detection, which works analogous to weak supervision but constructs an explicit likelihood ratio using normalizing flows instead of training a classifier.
- `demos/tree_classifier.ipynb` discusses the challenge of uninformative features in weakly supervised anomaly detection and demonstrates substantial improvement by using ensembles of boosted decision trees instead of neural networks.

## Installation

Just clone via the usual way. The `requirements.txt` covers the necessary libraries to run the default demo notebook.

In order to run the Normalizing Flow with Pyro backend, one further needs to install [Pyro](https://pyro.ai/). For the Conditional Flow Matching generative model, one needs to install [torchdyn](https://torchdyn.org/). The `requirements_full.txt` includes these additional dependencies.

## Overview of existing building blocks

Below is a brief overview of the building blocks provided in the `sk_cathode/` directory, out of which one can build anomaly detection pipelines.

### Generative models

- `sk_cathode.generative_models.conditional_normalizing_flow_torch.ConditionalNormalizingFlow`: A conditional normalizing flow model  implemented directly in PyTorch.
- `sk_cathode.generative_models.conditional_normalizing_flow_pyro.ConditionalNormalizingFlow`: A conditional normalizing flow model  implemented via the Pyro library.
- `sk_cathode.generative_models.conditional_normalizing_flow_nflows.ConditionalNormalizingFlow`: A conditional normalizing flow model  implemented via nFlows.
- `sk_cathode.generative_models.conditional_flow_matching.ConditionalFlowMatching`: A conditional flow matching model  implemented via PyTorch and the torchdyn ODE solver.

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

## Contributing

The repo is very much open for contributions via pull requests, e.g. to add more classifiers and generative models, other ways of doing anomaly detection, or just an illustrative demo notebook highlighting a particular study. For questions, feel free to contact `manuel.sommerhalder[at]uni-hamburg.de`.
