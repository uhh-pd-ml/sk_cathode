import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from os import makedirs
from os.path import join
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm


class AutoencoderModel(nn.Module):
    """A PyTorch module implementing a simple feed-forward neural network.
    """

    def __init__(self, layers=[32, 16, 4, 16, 32], n_inputs=10):
        super().__init__()

        self.layers = []
        for i, nodes in enumerate(layers):
            self.layers.append(nn.Linear(n_inputs, nodes))

            # Don't add activation function for last layer
            if (i < (len(layers) - 1)):
                self.layers.append(nn.ReLU())
            n_inputs = nodes

        self.model_stack = nn.Sequential(*self.layers)

    def forward(self, X):
        return self.model_stack(X)


class Autoencoder(BaseEstimator):
    """An autoencoder based on torch but wrapped such that it
    mimicks the scikit-learn API, using numpy arrays as inputs and outputs.

    Parameters
    ----------
    save_path : str, optional
        Path to save the model to. If None, no model is saved.
        If provided, the model will use the best checkpoint after training.
    load : bool, optional
        Whether to load the model from save_path.
    n_inputs : int, default=4
        Number of input features.
    layers : list, default=[8, 32, 16, 2, 16, 32, 8]
        List of integers, specifying the number of nodes in each layer.
    lr : float, default=0.001
        Learning rate during training.
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

    def __init__(
            self,
            save_path=None,
            load=False,
            n_inputs=8,
            layers=[
                8,
                32,
                16,
                2,
                16,
                32,
                8],
            lr=0.001,
            early_stopping=False,
            patience=10,
            no_gpu=False,
            val_split=0.2,
            batch_size=128,
            epochs=100,
            verbose=False):

        self.save_path = save_path
        if save_path is not None:
            self.clsf_model_path = join(save_path, "AE_models/")
        else:
            self.clsf_model_path = None

        self.layers = layers
        self.n_inputs = n_inputs
        self.lr = lr

        self.model = AutoencoderModel(layers, n_inputs=n_inputs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss = F.mse_loss
        self.no_gpu = no_gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   and not no_gpu else "cpu")
        self.early_stopping = early_stopping
        self.patience = patience
        self.val_split = val_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.model.to(self.device)

        # defaulting to eval mode, switching to train mode in fit()
        self.model.eval()

        self.load = load

        if load:
            self.load_best_model()

    def predict_proba(self, X):
        """Runs input data through the model and computes reconstruction error (MSE loss) which
        can be used as an anomaly score

        Parameters
        ----------
        X : numpy.ndarray
            Input data.

        Returns
        -------
        reco_error : numpy.ndarray
            Reconstruction erorr (MSE loss) on each example
        """

        reco = self.transform(X)
        reco_error = np.sum((X - reco)**2, axis=-1)

        return reco_error

    def transform(self, X):
        """Compresses and decompresses the input data

        Parameters
        ----------
        X : numpy.ndarray
            Input data.

        Returns
        -------
        prediction : numpy.ndarray
            Output array
        """
        with torch.no_grad():
            self.model.eval()
            X = torch.from_numpy(X).type(torch.FloatTensor).to(self.device)
            prediction = self.model.forward(X).detach().cpu().numpy()
        return prediction

    def fit(self, X, X_val=None,
            sample_weight=None, sample_weight_val=None):
        """Fits (trains) the model to the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        X_val : numpy.ndarray, optional
            Validation input data.
        sample_weight : numpy.ndarray, optional (Not yet implemented!)
            Sample weights for the training data.
        sample_weight_val : numpy.ndarray, optional  (Not yet implemented!)
            Sample weights for the validation data.

        Returns
        -------
        self : object
            An instance of the classifier.
        """

        assert not (self.epochs is None and not self.early_stopping), (
            "A finite number of epochs must be set if early stopping"
            " is not used!")

        assert (sample_weight is None and sample_weight_val is None), (
            "Sample weights for autoencoder training not yet implemented!")

        # allowing not to provide validation set, just for compatibility with
        # the sklearn API
        if X_val is None:
            if self.val_split is None or not (self.val_split > 0.
                                              and self.val_split < 1.):
                raise ValueError("val_split is needs to be provided and lie "
                                 "between 0 and 1 in case X_val is "
                                 "not provided!")
            else:
                X_train, X_val = train_test_split(
                    X, test_size=self.val_split, shuffle=True)
        else:
            X_train = X.copy()

        if self.clsf_model_path is not None:
            makedirs(self.clsf_model_path, exist_ok=True)

        nan_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[nan_mask]

        nan_mask = ~np.isnan(X_val).any(axis=1)
        X_val = X_val[nan_mask]

        # build data loader out of numpy arrays

        X_train_torch = torch.from_numpy(
            X).type(torch.FloatTensor).to(self.device)
        X_train_dataset = torch.utils.data.TensorDataset(X_train_torch)
        train_loader = torch.utils.data.DataLoader(
            X_train_dataset, batch_size=self.batch_size, shuffle=True)

        X_val_torch = torch.from_numpy(
            X).type(torch.FloatTensor).to(self.device)
        X_val_dataset = torch.utils.data.TensorDataset(X_val_torch)
        val_loader = torch.utils.data.DataLoader(
            X_val_dataset, batch_size=self.batch_size, shuffle=True)

        # training loop
        self.model.train()
        for epoch in range(self.epochs if self.epochs is not None else 10000):
            print('\nEpoch: {}'.format(epoch))
            pbar = tqdm(total=len(train_loader.dataset))
            epoch_train_loss = 0.
            epoch_val_loss = 0.

            for i, batch in enumerate(train_loader):

                batch_inputs = batch[0]
                batch_inputs = batch_inputs.to(self.device)

                self.optimizer.zero_grad()
                batch_outputs = self.model(batch_inputs)
                batch_loss = self.loss(batch_outputs, batch_inputs)
                batch_loss.backward()
                self.optimizer.step()
                epoch_train_loss += batch_loss.item()
                if self.verbose:
                    pbar.update(batch_inputs.size(0))
                    pbar.set_description(
                        "Train loss: {:.6f}".format(
                            epoch_train_loss / (i + 1)))

            epoch_train_loss /= (i + 1)
            if self.verbose:
                pbar.close()

            with torch.no_grad():
                self.model.eval()
                for i, batch in enumerate(val_loader):

                    batch_inputs = batch[0]

                    batch_inputs = batch_inputs.to(self.device)

                    batch_outputs = self.model(batch_inputs)
                    batch_loss = self.loss(batch_outputs, batch_inputs)
                    epoch_val_loss += batch_loss.item()
                epoch_val_loss /= (i + 1)

            print("Validation loss:", epoch_val_loss)

            if epoch == 0:
                train_losses = np.array([epoch_train_loss])
                val_losses = np.array([epoch_val_loss])
            else:
                train_losses = np.concatenate(
                    (train_losses, np.array([epoch_train_loss])))
                val_losses = np.concatenate(
                    (val_losses, np.array([epoch_val_loss])))

            if self.save_path is not None:
                np.save(self._train_loss_path(),
                        train_losses)
                np.save(self._val_loss_path(),
                        val_losses)
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

    def load_best_model(self):
        """Loads the best model state from the provided save_path.
        """
        val_losses = self.load_val_loss()
        best_epoch = np.argmin(val_losses)
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
        return join(self.save_path, "CLSF_train_losses.npy")

    def _val_loss_path(self):
        return join(self.save_path, "CLSF_val_losses.npy")

    def _model_path(self, epoch):
        return join(self.clsf_model_path, f"CLSF_epoch_{epoch}.par")
