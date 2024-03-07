# wrapping the BDT classifer in a sklearn-like API
# heavily inspired by Tobias Quadfasel's work
# at https://github.com/uhh-pd-ml/treebased_ad

import joblib
import numpy as np
from copy import deepcopy
from os import makedirs
from os.path import join
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import class_weight


class HGBClassifier(BaseEstimator):
    """HistGradientBoosting tree classifier based on torch but wrapped such
    that it mimicks the scikit-learn API, using numpy arrays as inputs
    and outputs. In addition to the explicit parameters, more arguments can
    be passed, specific to the sklearn HistGradientBoostingClassifier.

    Parameters
    ----------
    save_path : str, optional
        Path to save the model to. If None, no model is saved.
        If provided, the model will use the best checkpoint after training.
    load : bool, optional
        Whether to load the model from save_path.
    max_bins : int, default=127
        Maximum number of bins to use for discretization.
    early_stopping : bool, default=False
        Whether to use early stopping. If set, the provided number of
        iterations will be treated as an upper limit.
    patience : int, default=10
        Number of iterations to wait for improvement before stopping, if early
        stopping is used.
    val_split : float, default=0.2
        Fraction of the training set to use for validation. Only has an
        effect if no validation set is provided to the fit method.
    max_iter : int, default=100
        Number of iterations to train for. In case early stopping is used,
        this is treated as an upper limit. Then also None can be provided,
        in which case the training will continue until early stopping
        is triggered.
    split_seed : int, optional
        Seed for the under-the-hood validation split of the tree.
    use_class_weights : bool, default=True
        Whether to use class weights during training.
    verbose : bool, default=False
        Whether to print progress during training.
    """

    def __init__(self, *args, save_path=None, load=False,
                 max_bins=127,  early_stopping=True,
                 patience=10, max_iters=100, val_split=0.2,
                 split_seed=None, use_class_weights=True, verbose=False,
                 **kwargs):

        self.save_path = save_path
        if save_path is not None:
            self.clsf_model_path = join(save_path, "CLSF_models/")
        else:
            self.clsf_model_path = None

        self.max_iters = max_iters
        self.early_stopping = early_stopping
        self.load = load
        self.max_bins = max_bins
        self.patience = patience
        self.val_split = val_split
        self.split_seed = split_seed
        self.use_class_weights = use_class_weights
        self.verbose = verbose
        self.args = args
        self.kwargs = kwargs

        self.model = HistGradientBoostingClassifier(
            *args, max_bins=max_bins, class_weight="balanced",
            max_iter=1, early_stopping=False, warm_start=True,
            **kwargs)

        if split_seed is not None:
            self.model.split_seed = split_seed

        if load:
            self.load_best_model()

    def fit(self, X, y, X_val=None, y_val=None,
            sample_weight=None, sample_weight_val=None):
        """Fits (trains) the model to the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Target data.
        X_val : numpy.ndarray, optional
            Validation input data.
        y_val : numpy.ndarray, optional
            Validation target data.
        sample_weight : numpy.ndarray, optional
            Sample weights for the training data.
        sample_weight_val : numpy.ndarray, optional
            Sample weights for the validation data.

        Returns
        -------
        self : object
            An instance of the classifier.
        """

        assert not (self.max_iters is None and not self.early_stopping), (
            "A finite number of iterations must be set if early stopping"
            " is not used!")

        # allowing not to provide validation set, just for compatibility with
        # the sklearn API
        if X_val is None and y_val is None:
            if self.val_split is None or not (self.val_split > 0.
                                              and self.val_split < 1.):
                raise ValueError("val_split is needs to be provided and lie "
                                 "between 0 and 1 in case X_val and y_val are "
                                 "not provided!")
            else:
                if sample_weight is not None:
                    (X_train, X_val, y_train, y_val,
                     sample_weight_train, sample_weight_val
                     ) = train_test_split(
                        X, y, sample_weight, test_size=self.val_split,
                        shuffle=True)
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=self.val_split, shuffle=True)
        else:
            X_train = X.copy()
            y_train = y.copy()
            if sample_weight is not None:
                sample_weight_train = sample_weight.copy()
            else:
                sample_weight_train = None

        if self.clsf_model_path is not None:
            makedirs(self.clsf_model_path, exist_ok=True)

        nan_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[nan_mask]
        y_train = y_train[nan_mask]
        if sample_weight_train is not None:
            sample_weight_train = sample_weight_train[nan_mask]

        nan_mask = ~np.isnan(X_val).any(axis=1)
        X_val = X_val[nan_mask]
        y_val = y_val[nan_mask]
        if sample_weight_val is not None:
            sample_weight_val = sample_weight_val[nan_mask]

        # deduce class weights for training and validation sets
        # and translate them to sample weights
        # (move outside class as in sklearn?)
        if self.use_class_weights:
            class_weights_train = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train)
            class_weights_train = dict(enumerate(class_weights_train))
            sample_weight_train_ = class_weight_to_sample_weight(
                y_train, class_weights_train)

            if sample_weight_train is None:
                sample_weight_train = sample_weight_train_
            else:
                sample_weight_train = sample_weight.copy()
                sample_weight_train *= sample_weight_train_

            class_weights_val = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_val), y=y_val)
            class_weights_val = dict(enumerate(class_weights_val))
            sample_weight_val_ = class_weight_to_sample_weight(
                y_val, class_weights_val)

            if sample_weight_val is None:
                sample_weight_val = sample_weight_val_
            else:
                sample_weight_val = sample_weight_val.copy()
                sample_weight_val *= sample_weight_val_

        min_val_loss = np.inf
        train_losses = []
        val_losses = []
        for i in range(self.max_iters
                       if self.max_iters is not None else 10000):
            if self.verbose:
                print(f"training iteration {i}...")
            self.model.fit(X, y)

            train_preds = self.model.predict_proba(X)[:, 1]
            train_loss = log_loss(y, train_preds,
                                  sample_weight=sample_weight_train)

            val_preds = self.model.predict_proba(X_val)[:, 1]
            val_loss = log_loss(y_val, val_preds,
                                sample_weight=sample_weight_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if self.verbose:
                print(f"\ttrain loss: {train_loss}, val loss: {val_loss}")

            if val_loss < min_val_loss-1e-7:
                min_val_loss = val_loss
                iter_diff = 0
                self.model.best_model_state = deepcopy(self.model)
                self.model.best_iter = i
            else:
                iter_diff += 1

            if self.early_stopping and (iter_diff >= self.patience):
                print("Early stopping at iteration", i)
                break

            self.model.max_iter += 1

        self.model.train_losses = train_losses
        self.model.val_losses = val_losses

        # in contrast to the NN, all iterations are saved in a single file
        if self.save_path is not None:
            self._save_model(self._model_path())
            np.save(self._train_loss_path(), train_losses)
            np.save(self._val_loss_path(), val_losses)

            print("Loading best model state...")
            self.load_best_model()

        return self

    def predict(self, X):
        """Predicts the class labels for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.

        Returns
        -------
        prediction : numpy.ndarray
            Predicted class labels.
        """
        # The sklearn BDT predicts integer class labels. However, here we work
        # with binary classification and we want the freedom to choose our own
        # working point. Thus, let's return the 1-class probabilities instead.
        return self.model.predict_proba(X)[:, 1:2]

    def predict_proba(self, X):
        """Returns the class labels  probabilities for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.

        Returns
        -------
        prediction : numpy.ndarray
            Predicted class probablities.
        """
        return self.model.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.

        y : numpy.ndarray
            Target data.

        sample_weight : numpy.ndarray, optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        return self.model.score(X, y, sample_weight=sample_weight)

    def load_best_model(self):
        """Loads the best model state from the provided save_path.
        """
        self._load_model(self._model_path())
        self.model = self.model.best_model_state

    def load_train_loss(self):
        """Loads the training loss from the provided save_path.

        Returns
        -------
        train_loss : numpy.ndarray
            Training loss.
        """
        if self.save_path is None:
            raise ValueError("save_path is None, cannot load train loss")
        return np.load(self._train_loss_path())

    def load_val_loss(self):
        """Loads the validation loss from the provided save_path.

        Returns
        -------
        val_loss : numpy.ndarray
            Validation loss.
        """
        if self.save_path is None:
            raise ValueError("save_path is None, cannot load val loss")
        return np.load(self._val_loss_path())

    def _load_model(self, model_path):
        self.model = joblib.load(model_path)

    def _save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def _train_loss_path(self):
        return join(self.save_path, "CLSF_train_losses.npy")

    def _val_loss_path(self):
        return join(self.save_path, "CLSF_val_losses.npy")

    def _model_path(self):
        return join(self.clsf_model_path, "CLSF_model.joblib")


def class_weight_to_sample_weight(y, class_weights):
    """Converts class weights to sample weights.

    Parameters
    ----------
    y : numpy.array
        Target data.
    class_weights : dict
        Class weights.

    Returns
    -------
    sample_weights : numpy.array
        Sample weights.
    """

    return ((np.ones(y.shape) - y)
            * class_weights[0] + y*class_weights[1])
