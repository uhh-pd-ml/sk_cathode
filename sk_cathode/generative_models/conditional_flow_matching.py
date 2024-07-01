# wrapping conditional flow matching in a sklearn-like API
import numpy as np
import sklearn
import time
import torch
import torch.nn as nn
import torch.optim as optim

from os import makedirs
from os.path import join
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torchdyn.core import NeuralODE
from tqdm import tqdm


sklearn.set_config(enable_metadata_routing=True)


class ConditionalFlowMatching(BaseEstimator):
    """Conditional Flow Matching Model wrapped such that it
    mimicks the scikit-learn API, using numpy arrays as inputs and outputs.

    Parameters
    ----------
    save_path : str, optional
        Path to save the model to. If None, no model is saved.
        If provided, the model will use the best checkpoint after training.
    load : bool, optional
        Whether to load the model from save_path.
    optimizer : str, default="Adam"
        Type of the optimizer. Currently only "Adam" is implemented.
    num_inputs : int, default=4
        Number of inputs to the model. These are the ones being modeled.
    num_cond_inputs : int, default=1
        Number of conditional inputs to the model.
        BEWARE: currently only 1 is supported.
    num_blocks : int, default=1
        Number of blocks in the model.
    activation_function : str, default="ELU"
        Activation function to use.
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
    drop_last : bool, default=True
        Whether to drop the last training batch if it is smaller
        than the batch size.
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
        optimizer="Adam",
        num_inputs=4,
        num_cond_inputs=1,
        num_blocks=1,
        activation_function="ELU",
        lr=0.0001,
        weight_decay=0.000001,
        early_stopping=False,
        patience=10,
        no_gpu=False,
        val_split=0.2,
        batch_size=256,
        drop_last=True,
        epochs=100,
        verbose=False
    ):
        if optimizer != "Adam":
            raise NotImplementedError

        self.save_path = save_path
        if save_path is not None:
            self.de_model_path = join(save_path, "DE_models/")
        else:
            self.de_model_path = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   and not no_gpu else "cpu")
        self.num_inputs = num_inputs
        self.early_stopping = early_stopping
        self.patience = patience
        self.val_split = val_split
        self.batch_size = batch_size
        self.drop_last = drop_last
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

        flows = nn.ModuleList()
        for _ in range(num_blocks):
            flows.append(CNF(num_inputs, freqs=3,
                             activation=activation_function))
        self.model = flows

        self.model.to(self.device)
        total_parameters = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        print(f"ConditionalFlowMatching has {total_parameters} parameters")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr,
                                    weight_decay=weight_decay)

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
        m : numpy.ndarray
            Conditional data.
        X_val : numpy.ndarray, optional
            Validation input data.
        m_val : numpy.ndarray, optional
            Validation conditional data.

        Returns
        -------
        self : object
            An instance of the classifier.
        """

        assert not (self.epochs is None and not self.early_stopping), (
            "A finite number of epochs must be set if early stopping"
            " is not used!")

        assert len(X) >= self.batch_size, (
            "Batch size needs to be smaller than the number of samples!")

        # allowing not to provide validation set, just for compatibility with
        # the sklearn API
        if X_val is None and m_val is None:
            if self.val_split is None or not (self.val_split > 0.0
                                              and self.val_split < 1.0):
                raise ValueError(
                    "val_split is needs to be provided and lie "
                    "between 0 and 1 in case X_val and m_val are "
                    "not provided!"
                )
            else:
                X_train, X_val, m_train, m_val = train_test_split(
                    X, m, test_size=self.val_split, shuffle=True
                )
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
        train_loader = numpy_to_torch_loader(
            X_train, m_train, batch_size=self.batch_size, shuffle=True,
            drop_last=self.drop_last, device=self.device
        )
        val_loader = numpy_to_torch_loader(
            X_val, m_val, batch_size=self.batch_size, shuffle=True,
            drop_last=False, device=self.device
        )

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
            print("\nEpoch: {}".format(epoch))

            train_loss = train_epoch(
                self.model, self.optimizer, train_loader,
                device=self.device, verbose=self.verbose
            )[0]
            val_loss = compute_loss_over_batches(self.model, val_loader,
                                                 device=self.device)[0]
            if np.isnan(val_loss):
                raise ValueError("Training yields NaN validation loss!")

            print("train_loss = ", train_loss)
            print("val_loss = ", val_loss)
            train_losses = np.concatenate((train_losses,
                                           np.array([train_loss])))
            val_losses = np.concatenate((val_losses, np.array([val_loss])))

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
        m : numpy.ndarray
            Conditional data.
        X_val : numpy.ndarray, optional
            Validation input data.
        m_val : numpy.ndarray, optional
            Validation conditional data.

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

        X_torch = torch.from_numpy(X).type(torch.FloatTensor).to(self.device)
        m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(self.device)
        Xt = self.model(X_torch, m_torch)[0]

        return Xt.cpu().detach().numpy()

    def inverse_transform(self, Xt, m=None):
        raise NotImplementedError(
            "inverse_transform not implemented")

    def log_jacobian_determinant(self, X, m=None):
        raise NotImplementedError(
            "log_jacobian_determinant not implemented")

    def jacobian_determinant(self, X, m=None):
        raise NotImplementedError(
            "jacobian_determinant not implemented")

    def inverse_jacobian_determinant(self, X, m=None):
        raise NotImplementedError(
            "inverse_jacobian_determinant not implemented")

    def inverse_log_jacobian_determinant(self, X, m=None):
        raise NotImplementedError(
            "log_inverse_jacobian_determinant not implemented")

    def _sample(self, n_samples=1, cond=None, ode_solver="midpoint",
                ode_steps=100):
        """Sampling without batched generation.
        Might create memory issues for large n_samples."""

        samples = torch.randn(n_samples, self.num_inputs).to(self.device)
        cond = torch.tensor(cond).to(self.device)
        for f in reversed(self.model):
            samples = f.decode(
                samples,
                cond=cond,
                ode_solver=ode_solver,
                ode_steps=ode_steps,
            )

        return samples.cpu().detach().numpy()

    def sample(self, n_samples=1, m=None,
               solver_config={"solver": "midpoint", "steps": 100}):
        """Samples from the model.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw.
        m : numpy.ndarray
            Conditional data. Needs to be provided.
        solver_config : dict, default={"solver": "midpoint", "steps": 100}
            Configuration for the ODE solver.

        Returns
        -------
        X : numpy.ndarray
            Samples from the model.
        """
        if m is None:
            raise ValueError("m needs to be provided!")

        cond = torch.tensor(m, dtype=torch.float32).to(self.device)

        samples = generate_data(
            self,
            n_samples,
            cond=cond,
            ode_solver=solver_config["solver"],
            ode_steps=solver_config["steps"],
        )[0]

        return samples

    def predict_log_proba(self, X, m=None):
        raise NotImplementedError("predict_log_proba not implemented")

    def predict_proba(self, X, m=None):
        raise NotImplementedError("predict_proba not implemented")

    def score_samples(self, X, m=None):
        raise NotImplementedError("score_samples not implemented")

    def score(self, X, m=None):
        raise NotImplementedError("score not implemented")

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
                          drop_last=False, device=torch.device("cpu")):
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
    drop_last : bool, default=False
        Whether to drop the last batch if it is smaller than the batch size.
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
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             drop_last=drop_last,
                                             shuffle=shuffle)
    return dataloader


def compute_loss_over_batches(model, data_loader,
                              device=torch.device("cpu")):
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

            loss_func = FlowMatchingLoss(model, sigma=1e-4)

            loss = loss_func(data, cond=cond_data)

            now_loss += loss.item()
            end_loss = now_loss / (batch_idx + 1)
        print("n_nans =", n_nans)
        print("n_highs =", n_highs)
        return (end_loss,)


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
    loss_func = FlowMatchingLoss(model, sigma=1e-4)

    model.train()
    train_loss = 0
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

        train_loss = loss_func(data, cond=cond_data)

        train_loss.backward()
        optimizer.step()

        if verbose:
            pbar.update(data.size(0))
            pbar.set_description("Train Loss: {:.6f}".format(train_loss))

    if verbose:
        pbar.close()

    return (train_loss.item(),)


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: list[int] = [64, 64],
        activation: str = "ELU",
    ):
        layers = []

        for a, b in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            layers.extend([nn.Linear(a, b), getattr(nn, activation)()])

        super().__init__(*layers[:-1])


class small_cond_MLP_model(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "ELU",
        dim_t: int = 6,
        dim_cond: int = 1,
    ):
        super().__init__()
        self.mlp1 = MLP(
            in_features + dim_t + dim_cond,
            out_features=64,
            hidden_features=[64, 64],
            activation=activation,
        )
        self.mlp2 = MLP(
            64 + dim_t + dim_cond,
            out_features=256,
            hidden_features=[256, 256],
            activation=activation,
        )
        self.mlp3 = MLP(
            256 + dim_t + dim_cond,
            out_features=256,
            hidden_features=[256, 256],
            activation=activation,
        )
        self.mlp4 = MLP(
            256 + dim_t + dim_cond,
            out_features=out_features,
            hidden_features=[64, 64],
            activation=activation,
        )

    def forward(self, t, x, cond):
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp1(x)
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp2(x)
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp4(x)
        return x


def generate_data(
    model,
    num_jet_samples: int,
    batch_size: int = 256,
    cond: torch.Tensor = None,
    verbose: bool = True,
    ode_solver: str = "midpoint",
    ode_steps: int = 100,
):
    """Generate data with a model in batches and measure time.

    Args:
        model (_type_): Model with sample method
        num_jet_samples (int): Number of jet samples to generate
        batch_size (int, optional): Batch size for generation. Defaults to 256.
        cond (torch.Tensor, optional): Conditioned data if model
            is conditioned. Defaults to None.
        verbose (bool, optional): Print generation progress. Defaults to True.
        ode_solver (str, optional): ODE solver for sampling.
            Defaults to "dopri5_zuko".
        ode_steps (int, optional): Number of steps for ODE solver.
            Defaults to 100.


    Returns:
        np.array: sampled data of shape
            (num_jet_samples, num_particles, num_features)
            with features (eta, phi, pt)
        float: generation time
    """
    if verbose:
        print(f"Generating data ({num_jet_samples} samples).")
    particle_data_sampled = []
    start_time = 0

    for i in tqdm(range(num_jet_samples // batch_size), disable=not verbose):
        if cond is not None:
            cond_batch = cond[i*batch_size:(i + 1)*batch_size]
        else:
            cond_batch = None
        if i == 1:
            start_time = time.time()

        with torch.no_grad():
            jet_samples_batch = model._sample(
                n_samples=batch_size,
                cond=cond_batch,
                ode_solver=ode_solver,
                ode_steps=ode_steps,
            )

        particle_data_sampled.append(jet_samples_batch)
    particle_data_sampled = np.concatenate(particle_data_sampled)

    end_time = time.time()

    if num_jet_samples % batch_size != 0:
        remaining_samples = num_jet_samples - (num_jet_samples //
                                               batch_size * batch_size)
        if cond is not None:
            cond_batch = cond[-remaining_samples:]
        else:
            cond_batch = None

        with torch.no_grad():
            jet_samples_batch = model._sample(
                n_samples=remaining_samples,
                cond=cond_batch,
                ode_solver=ode_solver,
                ode_steps=ode_steps,
            )

    particle_data_sampled = np.concatenate([particle_data_sampled,
                                            jet_samples_batch])
    generation_time = end_time - start_time
    return particle_data_sampled, generation_time


class FlowMatchingLoss(nn.Module):
    """Flow matching loss.

    introduced in: https://arxiv.org/abs/2210.02747

    Args:
        flows (nn.ModuleList): Module list of flows
        sigma (float, optional): Sigma. Defaults to 1e-4.
    """

    def __init__(self, flows: nn.ModuleList, sigma: float = 1e-4):
        super().__init__()
        self.flows = flows
        self.sigma = sigma

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None,
        cond: torch.Tensor = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x[..., 0]).unsqueeze(-1)

        if len(x.shape) == 3:
            # for set data
            t = torch.rand_like(torch.ones(x.shape[0]))
            t = t.unsqueeze(-1).repeat_interleave(x.shape[1], dim=1
                                                  ).unsqueeze(-1)
        else:
            t = torch.rand_like(x[..., 0]).unsqueeze(-1)
        t = t.type_as(x)

        z = torch.randn_like(x)

        y = (1 - t) * x + (self.sigma + (1 - self.sigma) * t) * z

        u_t = (1 - self.sigma) * z - x
        u_t = u_t * mask

        temp = y.clone()
        for v in self.flows:
            temp = v(t.squeeze(-1), temp, mask=mask, cond=cond)
        v_t = temp.clone()

        sqrd = (v_t - u_t).square()
        out = sqrd.sum() / mask.sum()  # mean with ignoring masked values
        return out


class ode_wrapper(torch.nn.Module):
    """Wraps model to ode solver compatible format.

    Args:
        model (torch.nn.Module): Model to wrap.
        mask (torch.Tensor, optional): Mask. Defaults to None.
        cond (torch.Tensor, optional): Condition. Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        mask: torch.Tensor = None,
        cond: torch.Tensor = None,
    ):
        super().__init__()
        self.model = model
        self.mask = mask
        self.cond = cond

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x, mask=self.mask, cond=self.cond)


class CNF(nn.Module):
    def __init__(
        self,
        features: int,
        freqs: int = 3,
        activation: str = "Tanh",
    ):
        super().__init__()
        # Any architecture can be used here
        self.net = small_cond_MLP_model(
            features, features, dim_t=2 * freqs, dim_cond=1,
            activation=activation
        )

        self.register_buffer("freqs", torch.arange(1, freqs + 1) * torch.pi)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        cond: torch.Tensor = None,
    ) -> torch.Tensor:
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(*x.shape[:-1], -1)

        return self.net(t, x, cond=cond)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        node = NeuralODE(x, solver="midpoint", sensitivity="adjoint")
        t_span = torch.linspace(0.0, 1.0, 50)
        traj = node.trajectory(x, t_span)
        return traj[-1]

    def decode(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor = None,
        ode_solver: str = "midpoint",
        ode_steps: int = 100,
    ) -> torch.Tensor:
        wrapped_cnf = ode_wrapper(
            model=self,
            mask=mask,
            cond=cond,
        )
        if ode_solver == "midpoint":
            node = NeuralODE(wrapped_cnf, solver="midpoint",
                             sensitivity="adjoint")
            t_span = torch.linspace(1.0, 0.0, ode_steps)
            traj = node.trajectory(z, t_span)
            return traj[-1]
        else:
            raise NotImplementedError(f"Solver {ode_solver} not implemented")

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("log_prob not implemented")
