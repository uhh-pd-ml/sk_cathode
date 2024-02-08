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

## Installation

Just clone via the usual way. The `requirements.txt` covers the necessary libraries to run the default demo notebook. In order to run the Normalizing Flow with Pyro backend, one further needs to install [Pyro](https://pyro.ai/). For the Conditional Flow Matching generative model, one needs to install [torchdyn](https://torchdyn.org/).

## Contributing

The repo is very much open for contributions via pull requests, e.g. to add more classifiers and generative models, other ways of doing anomaly detection, or just an illustrative demo notebook highlighting a particular study. For questions, feel free to contact `manuel.sommerhalder[at]uni-hamburg.de`.
