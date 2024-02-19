# wrapping the density estimator in a sklearn-like API, with nFlows as backend
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim

from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform  # noqa
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow

from os import makedirs
from os.path import join
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from tqdm import tqdm


sklearn.set_config(enable_metadata_routing=True)


class ConditionalNormalizingFlow(BaseEstimator):
    """Conditional normalizing flow based on nFlows but wrapped such that it
    mimicks the scikit-learn API, using numpy arrays as inputs and outputs.
    Currently only supports RQS MAFs.

    Parameters
    ----------
    save_path : str, optional
        Path to save the model to. If None, no model is saved.
        If provided, the model will use the best checkpoint after training.
    load : bool, optional
        Whether to load the model from save_path.
    model_type : str, default="MAF"
        Type of the model. Currently only "MAF" is implemented.
    optimizer : str, default="Adam"
        Type of the optimizer. Currently only "Adam" is implemented.
    num_inputs : int, default=4
        Number of inputs to the model. These are the ones being modeled.
    num_cond_inputs : int, default=1
        Number of conditional inputs to the model.
    num_blocks : int, default=10
        Number of MADE blocks in the model.
    num_hidden : int, default=64
        Number of hidden units in each MADE block.
    num_layers : int, default=2
        Number of layers in each MADE block.
    num_bins : int, default=8
        Number of bins in the piecewise rational quadratic spline.
    tail_bound : int, default=10
        Bound for the tails of the piecewise rational quadratic spline.
    batch_norm : bool, default=False
        Whether to use batch normalization.
    lr : float, default=0.0001
        Learning rate for the optimizer.
    weight_decay : float, default=0.000001
        Weight decay for the optimizer.
    early_stopping : bool, default=False
        Whether to use early stopping. If set, the provided number of
        epochs will be treated as an upper limit.
    patience : int, default=10
        Number of epochs to wait for improvement before stopping, if early
        stopping is used.
    no_gpu : bool, default=False
        Turns off GPU usages. By default the GPU is used if available.
    val_split : float, default=0.2
        Fraction of the training set to use for validation. Only has an
        effect if no validation set is provided to the fit method.
    batch_size : int, default=128
        Batch size during training.
    epochs : int, default=100
        Number of epochs to train for. In case early stopping is used,
        this is treated as an upper limit. Then also None can be provided,
        in which case the training will continue until early stopping
        is triggered.
    verbose : bool, default=False
        Whether to print progress during training.
    """
    def __init__(self, save_path=None, load=False,
                 model_type="MAF", optimizer="Adam",
                 num_inputs=4, num_cond_inputs=1, num_blocks=10,
                 num_hidden=64, num_layers=2, num_bins=8,
                 tail_bound=10, batch_norm=False, lr=0.0001,
                 weight_decay=0.000001, early_stopping=False,
                 patience=10, no_gpu=False,
                 val_split=0.2, batch_size=256, epochs=100, verbose=False):

        if model_type != "MAF":
            raise NotImplementedError
        if optimizer != "Adam":
            raise NotImplementedError

        self.save_path = save_path
        if save_path is not None:
            self.de_model_path = join(save_path, "DE_models/")
        else:
            self.de_model_path = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   and not no_gpu else "cpu")
        self.early_stopping = early_stopping
        self.patience = patience
        self.val_split = val_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        # metadata routing for pipeline
        # TODO implement same for jacobians and sampling
        self.set_fit_request(X_val=True, m_val=True)
        self.set_transform_request(m=True)
        self.set_inverse_transform_request(m=True)
        self.set_predict_log_proba_request(m=True)
        self.set_predict_proba_request(m=True)
        self.set_score_request(m=True)
        base_dist = StandardNormal(shape=[num_inputs])

        modules = []
        for i in range(num_blocks):
            modules += [
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    num_bins=num_bins,
                    tails='linear',
                    tail_bound=tail_bound,
                    features=num_inputs,
                    context_features=num_cond_inputs,
                    hidden_features=num_hidden,
                    num_blocks=num_layers,
                    random_mask=False,
                    activation=nn.ReLU(),
                    dropout_probability=0.,
                    use_batch_norm=batch_norm,
                    use_residual_blocks=False)
                ]
            modules += [ReversePermutation(features=num_inputs)]

        transform = CompositeTransform(modules)
        self.model = Flow(transform, base_dist)

        self.model.to(self.device)
        total_parameters = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        print(f"ConditionalNormalizingFlow has {total_parameters} parameters")

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

        # defaulting to eval mode, switching to train mode in fit()
        self.model.eval()

        if load:
            self.load_best_model()

    def fit(self, X, m, X_val=None, m_val=None):
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

        Returns
        -------
        self : object
            An instance of the classifier.
        """

        assert not (self.epochs is None and not self.early_stopping), (
            "A finite number of epochs must be set if early stopping"
            " is not used!")

        # allowing not to provide validation set, just for compatibility with
        # the sklearn API
        if X_val is None and m_val is None:
            if self.val_split is None or not (self.val_split > 0.
                                              and self.val_split < 1.):
                raise ValueError("val_split is needs to be provided and lie "
                                 "between 0 and 1 in case X_val and m_val are "
                                 "not provided!")
            else:
                X_train, X_val, m_train, m_val = train_test_split(
                    X, m, test_size=self.val_split, shuffle=True)
        else:
            X_train = X.copy()
            m_train = m.copy()

        if self.de_model_path is not None:
            makedirs(self.de_model_path, exist_ok=True)

        nan_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[nan_mask]
        m_train = m_train[nan_mask]

        nan_mask = ~np.isnan(X_val).any(axis=1)
        X_val = X_val[nan_mask]
        m_val = m_val[nan_mask]

        # build data loader out of numpy arrays
        train_loader = numpy_to_torch_loader(X_train, m_train,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             device=self.device)
        val_loader = numpy_to_torch_loader(X_val, m_val,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           device=self.device)

        # record also untrained losses
        train_loss = compute_loss_over_batches(self.model, train_loader,
                                               device=self.device)[0]
        val_loss = compute_loss_over_batches(self.model, val_loader,
                                             device=self.device)[0]
        train_losses = np.array([train_loss])
        val_losses = np.array([val_loss])
        if self.save_path is not None:
            np.save(self._train_loss_path(), train_losses)
            np.save(self._val_loss_path(), val_losses)

        # training loop
        for epoch in range(self.epochs if self.epochs is not None else 1000):
            print('\nEpoch: {}'.format(epoch))

            train_loss = train_epoch(self.model, self.optimizer, train_loader,
                                     device=self.device,
                                     verbose=self.verbose)[0]
            val_loss = compute_loss_over_batches(self.model, val_loader,
                                                 device=self.device)[0]

            if np.isnan(val_loss):
                raise ValueError("Training yields NaN validation loss!")

            print("train_loss = ", train_loss)
            print("val_loss = ", val_loss)
            train_losses = np.concatenate(
                (train_losses, np.array([train_loss])))
            val_losses = np.concatenate(
                (val_losses, np.array([val_loss])))

            if self.save_path is not None:
                np.save(self._train_loss_path(), train_losses)
                np.save(self._val_loss_path(), val_losses)
                self._save_model(self._model_path(epoch))

            if self.early_stopping:
                if epoch > self.patience:
                    if np.all(val_losses[-self.patience:] >
                              val_losses[-self.patience - 1]):
                        print("Early stopping at epoch", epoch)
                        break

        self.model.eval()
        if self.save_path is not None:
            print("Loading best model state...")
            self.load_best_model()

        return self

    def fit_transform(self, X, m, X_val=None, m_val=None):
        """Trains and then transforms the provided data to the latent space.

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

        Returns
        -------
        Xt : numpy.ndarray
            Latent space representation of the input data.
        """
        return self.fit(X, m=m, X_val=X_val, m_val=m_val).transform(X, m=m)

    def transform(self, X, m=None):
        """Transforms the provided data to the latent space.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data. Needs to be provided.

        Returns
        -------
        Xt : numpy.ndarray
            Latent space representation of the input data.
        """

        # m needs to be provided, but trying to mimick the sklearn API here
        if m is None:
            raise ValueError("m needs to be provided!")

        with torch.no_grad():
            X_torch = torch.from_numpy(X).type(torch.FloatTensor).to(
                self.device)
            m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(
                self.device)
            Xt = self.model._transform.forward(X_torch, context=m_torch)[0]

        return Xt.cpu().detach().numpy()

    def inverse_transform(self, Xt, m=None):
        """Transforms the provided latent space data to data space.

        Parameters
        ----------
        Xt : numpy.ndarray
            Latent input data.
        m : numpy.ndarray
            Conditional data. Needs to be provided.

        Returns
        -------
        X : numpy.ndarray
            Data space representation of the (latent) input data.
        """

        # m needs to be provided, but trying to mimick the sklearn API here
        if m is None:
            raise ValueError("m needs to be provided!")

        with torch.no_grad():
            Xt_torch = torch.from_numpy(Xt).type(torch.FloatTensor).to(
                self.device)
            m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(
                self.device)
            X = self.model._transform.inverse(Xt_torch, context=m_torch)[0]

        return X.cpu().detach().numpy()

    def log_jacobian_determinant(self, X, m=None):
        """Computes the log jacobian determinant of the transformation.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data. Needs to be provided.

        Returns
        -------
        log_det : numpy.ndarray
            Log jacobian determinant of the transformation.
        """

        # m needs to be provided, but trying to mimick the sklearn API here
        if m is None:
            raise ValueError("m needs to be provided!")

        with torch.no_grad():
            X_torch = torch.from_numpy(X).type(torch.FloatTensor).to(
                self.device)
            m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(
                self.device)
            log_det = self.model._transform.forward(X_torch,
                                                    context=m_torch)[1]

        return log_det.cpu().detach().numpy().reshape(-1, 1)

    def jacobian_determinant(self, X, m=None):
        """Computes the jacobian determinant of the transformation.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data. Needs to be provided.

        Returns
        -------
        jac : numpy.ndarray
            Jacobian determinant of the transformation.
        """
        return np.exp(self.log_jacobian_determinant(X, m=m))

    def inverse_log_jacobian_determinant(self, Xt, m=None):
        """Computes the inverse log jacobian determinant of the transformation.

        Parameters
        ----------
        Xt : numpy.ndarray
            Input data (latent space).
        m : numpy.ndarray
            Conditional data. Needs to be provided.

        Returns
        -------
        log_det : numpy.ndarray
            Inverse log jacobian determinant of the transformation.
        """

        # m needs to be provided, but trying to mimick the sklearn API here
        if m is None:
            raise ValueError("m needs to be provided!")

        with torch.no_grad():
            Xt_torch = torch.from_numpy(Xt).type(torch.FloatTensor).to(
                self.device)
            m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(
                self.device)
            log_det = self.model._transform.inverse(Xt_torch,
                                                    context=m_torch)[1]

        return log_det.cpu().detach().numpy().reshape(-1, 1)

    def inverse_jacobian_determinant(self, Xt, m=None):
        """Computes the inverse jacobian determinant of the transformation.

        Parameters
        ----------
        Xt : numpy.ndarray
            Input data (latent space).
        m : numpy.ndarray
            Conditional data. Needs to be provided.

        Returns
        -------
        jac : numpy.ndarray
            Inverse jacobian determinant of the transformation.
        """
        return np.exp(self.inverse_log_jacobian_determinant(Xt, m=m))

    def sample(self, n_samples=1, m=None):
        """Samples from the model.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw.
        m : numpy.ndarray
            Conditional data. Needs to be provided.

        Returns
        -------
        X : numpy.ndarray
            Samples from the model.
        """

        # m needs to be provided, but trying to mimick the sklearn API here
        if m is None:
            raise ValueError("m needs to be provided!")

        with torch.no_grad():
            m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(
                self.device)
            X_torch = self.model.sample(1, context=m_torch).view(n_samples, -1)
        return X_torch.cpu().detach().numpy()

    def predict_log_proba(self, X, m=None):
        """Predicts the log probability of the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data. Needs to be provided.

        Returns
        -------
        log_prob : numpy.ndarray
            Log probability of the provided data.
        """

        # m needs to be provided, but trying to mimick the sklearn API here
        if m is None:
            raise ValueError("m needs to be provided!")

        with torch.no_grad():
            X_torch = torch.from_numpy(X).type(torch.FloatTensor).to(
                self.device)
            m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(
                self.device)
            log_prob = self.model.log_prob(X_torch, m_torch)
        return log_prob.detach().cpu().numpy().reshape(-1, 1)

    def predict_proba(self, X, m=None):
        """Predicts the probability of the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data. Needs to be provided.

        Returns
        -------
        prob : numpy.ndarray
            Probability of the provided data.
        """
        return np.exp(self.predict_log_proba(X, m=m))

    def score_samples(self, X, m=None):
        """Predicts the log probability of the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data. Needs to be provided.

        Returns
        -------
        log_prob : numpy.ndarray
            Log probability of the provided data.
        """
        return self.predict_log_proba(X, m=m)

    def score(self, X, m=None):
        """Predicts the total log probability of the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        m : numpy.ndarray
            Conditional data. Needs to be provided.

        Returns
        -------
        log_prob : float
            Total log probability of the provided data.
        """
        return self.score_samples(X, m=m).sum().item()

    def load_best_model(self):
        """Loads the best model state from the provided save_path.
        """
        val_losses = self.load_val_loss()
        best_epoch = np.argmin(val_losses) - 1  # includes pre-training loss
        self.load_epoch_model(best_epoch)
        self.model.eval()

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

    def load_epoch_model(self, epoch):
        """Loads the model state from the provided save_path at the
        specified epoch.

        Parameters
        ----------
        epoch : int
            Epoch at which to load the model state.
        """
        self._load_model(self._model_path(epoch))

    def _load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path,
                                              map_location=self.device))

    def _save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def _train_loss_path(self):
        return join(self.save_path, "DE_train_losses.npy")

    def _val_loss_path(self):
        return join(self.save_path, "DE_val_losses.npy")

    def _model_path(self, epoch):
        return join(self.de_model_path, f"DE_epoch_{epoch}.par")


# utility functions

def numpy_to_torch_loader(X, m, batch_size=256, shuffle=True,
                          device=torch.device("cpu")):
    """Builds a torch DataLoader from numpy arrays.

    Parameters
    ----------
    X : numpy.ndarray
        Input data.
    m : numpy.ndarray
        Conditional data.
    batch_size : int, default=128
        Batch size.
    shuffle : bool, default=True
        Whether to shuffle the data.
    device : torch.device, default=torch.device("cpu")
        Device to use.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the provided data.
    """
    X_torch = torch.from_numpy(X).type(torch.FloatTensor).to(device)
    m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(device)
    dataset = torch.utils.data.TensorDataset(X_torch, m_torch)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def compute_loss_over_batches(model, data_loader, device=torch.device("cpu")):
    """Computes the loss over the provided data.

    Parameters
    ----------
    model : torch.nn.Module
        Model to use.
    data_loader : torch.utils.data.DataLoader
        DataLoader for the provided data.
    device : torch.device, default=torch.device("cpu")
        Device to use.

    Returns
    -------
    loss : float
        Loss over the provided data.
    """
    model.eval()
    with torch.no_grad():
        now_loss = 0
        n_nans = 0
        n_highs = 0
        for batch_idx, batch_data in enumerate(data_loader):

            data = batch_data[0]
            data = data.to(device)
            cond_data = batch_data[1].float()
            cond_data = cond_data.to(device)

            loss_vals_raw = model.log_prob(data, cond_data)
            loss_vals = loss_vals_raw.flatten()

            n_nans += sum(torch.isnan(loss_vals)).item()
            n_highs += sum(torch.abs(loss_vals) >= 1000).item()
            loss_vals = loss_vals[~torch.isnan(loss_vals)]
            loss_vals = loss_vals[torch.abs(loss_vals) < 1000]
            loss = -loss_vals.mean()
            loss = loss.item()

            now_loss += loss
            end_loss = now_loss / (batch_idx + 1)
        print("n_nans =", n_nans)
        print("n_highs =", n_highs)
        return (end_loss, )


def train_epoch(model, optimizer, data_loader,
                device=torch.device("cpu"), verbose=True):
    """Trains the provided model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to use.
    optimizer : torch.optim.Optimizer
        Optimizer to use.
    data_loader : torch.utils.data.DataLoader
        DataLoader for the provided data.
    device : torch.device, default=torch.device("cpu")
        Device to use.
    verbose : bool, default=True
        Whether to print progress during training.

    Returns
    -------
    train_loss : float
        Loss over the provided data.
    """
    model.train()
    train_loss = 0
    train_loss_avg = []
    if verbose:
        pbar = tqdm(total=len(data_loader.dataset))
    for batch_idx, data in enumerate(data_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None
            data = data[0]
        data = data.to(device)

        optimizer.zero_grad()
        loss = -model.log_prob(data, cond_data)
        train_loss += loss.mean().item()
        train_loss_avg.extend(loss.tolist())
        loss.mean().backward()
        optimizer.step()

        if verbose:
            pbar.update(data.size(0))
            pbar.set_description(
                "Train, Log likelihood in nats: {:.6f}".format(
                    -train_loss / (batch_idx + 1)))

    if verbose:
        pbar.close()

    return (np.array(train_loss_avg).flatten().mean(), )
