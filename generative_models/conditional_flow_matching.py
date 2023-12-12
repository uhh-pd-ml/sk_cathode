# wrapping the density estimator in a sklearn-like API
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchdyn.core import NeuralODE

from os import makedirs
from os.path import join
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class ConditionalFlowMatching:
    """Conditional Flow Matching Model wrapped such that it
    mimicks the scikit-learn API, using numpy arrays as inputs and outputs.
    """

    def __init__(
        self,
        save_path=None,
        optimizer="Adam",
        num_inputs=4,
        num_cond_inputs=1,
        num_blocks=1,
        activation_function="ELU",
        lr=0.0001,
        weight_decay=0.000001,
        no_gpu=False,
    ):
        if optimizer != "Adam":
            raise NotImplementedError

        self.save_path = save_path
        self.de_model_path = join(save_path, "DE_models/")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not no_gpu else "cpu")
        self.num_inputs = num_inputs

        flows = nn.ModuleList()
        for _ in range(num_blocks):
            flows.append(CNF(num_inputs, freqs=3, activation=activation_function))
        self.model = flows

        self.model.to(self.device)
        total_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"ConditionalFlowMatching has {total_parameters} parameters")

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # defaulting to eval mode, switching to train mode in fit()
        self.model.eval()

    def fit(
        self,
        X,
        m,
        X_val=None,
        m_val=None,
        val_split=0.2,
        batch_size=256,
        epochs=100,
        verbose=False,
    ):
        # allowing not to provide validation set, just for compatibility with
        # the sklearn API
        if X_val is None and m_val is None:
            if val_split is None or not (val_split > 0.0 and val_split < 1.0):
                raise ValueError(
                    "val_split is needs to be provided and lie "
                    "between 0 and 1 in case X_val and m_val are "
                    "not provided!"
                )
            else:
                X_train, X_val, m_train, m_val = train_test_split(
                    X, m, test_size=val_split, shuffle=True
                )
        else:
            X_train = X.copy()
            m_train = m.copy()

        makedirs(self.de_model_path, exist_ok=True)

        nan_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[nan_mask]
        m_train = m_train[nan_mask]

        nan_mask = ~np.isnan(X_val).any(axis=1)
        X_val = X_val[nan_mask]
        m_val = m_val[nan_mask]

        # build data loader out of numpy arrays
        train_loader = numpy_to_torch_loader(
            X_train, m_train, batch_size=batch_size, shuffle=True, device=self.device
        )
        val_loader = numpy_to_torch_loader(
            X_val, m_val, batch_size=batch_size, shuffle=True, device=self.device
        )

        # record also untrained losses
        train_loss = compute_loss_over_batches(self.model, train_loader, device=self.device)[0]
        val_loss = compute_loss_over_batches(self.model, val_loader, device=self.device)[0]
        train_losses = np.array([train_loss])
        val_losses = np.array([val_loss])
        if self.save_path is not None:
            np.save(join(self.save_path, "DE_train_losses.npy"), train_losses)
            np.save(join(self.save_path, "DE_val_losses.npy"), val_losses)

        # training loop
        for epoch in range(epochs):
            print("\nEpoch: {}".format(epoch))

            train_loss = train_epoch(
                self.model, self.optimizer, train_loader, device=self.device, verbose=verbose
            )[0]
            val_loss = compute_loss_over_batches(self.model, val_loader, device=self.device)[0]

            print("train_loss = ", train_loss)
            print("val_loss = ", val_loss)
            train_losses = np.concatenate((train_losses, np.array([train_loss])))
            val_losses = np.concatenate((val_losses, np.array([val_loss])))

            if self.save_path is not None:
                np.save(join(self.save_path, "DE_train_losses.npy"), train_losses)
                np.save(join(self.save_path, "DE_val_losses.npy"), val_losses)
                self._save_model(join(self.de_model_path, f"DE_epoch_{epoch}.par"))

        self.model.eval()

    def transform(self, X, m=None):
        # m needs to be provided, but trying to mimick the sklearn API here
        if m is None:
            raise ValueError("m needs to be provided!")

        X_torch = torch.from_numpy(X).type(torch.FloatTensor).to(self.device)
        m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(self.device)
        Z = self.model(X_torch, m_torch)[0]

        return Z.cpu().detach().numpy()

    def _sample(self, n_samples=1, cond=None, ode_solver="midpoint", ode_steps=100):
        """Sampling without batched generation. Might create memory issues for large n_samples.

        Args:
            n_samples (int, optional): _description_. Defaults to 1.
            cond (_type_, optional): _description_. Defaults to None.
            solver_config (dict, optional): _description_. Defaults to {"solver": "midpoint", "steps": 100}.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

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

    def sample(self, n_samples=1, m=None, solver_config={"solver": "midpoint", "steps": 100}):
        # m needs to be provided, but trying to mimick the sklearn API here
        if m is None:
            raise ValueError("m needs to be provided!")

        samples = generate_data(
            self,
            n_samples,
            cond=m,
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
        if self.save_path is None:
            raise ValueError("save_path is None, cannot load best model")
        val_losses = np.load(join(self.save_path, "DE_val_losses.npy"))
        best_epoch = np.argmin(val_losses) - 1  # includes pre-training loss
        self._load_model(join(self.de_model_path, f"DE_epoch_{best_epoch}.par"))
        self.model.eval()

    def _load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def _save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)


# utility functions


def numpy_to_torch_loader(X, m, batch_size=256, shuffle=True, device=torch.device("cpu")):
    X_torch = torch.from_numpy(X).type(torch.FloatTensor).to(device)
    m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(device)
    dataset = torch.utils.data.TensorDataset(X_torch, m_torch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def compute_loss_over_batches(model, data_loader, device=torch.device("cpu")):
    # for computing the averaged loss over the entire dataset.
    # Mainly useful for tracking losses during training
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


def train_epoch(model, optimizer, data_loader, device=torch.device("cpu"), verbose=True):
    # Does one epoch of model training.

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
        cond (torch.Tensor, optional): Conditioned data if model is conditioned. Defaults to None.
        verbose (bool, optional): Print generation progress. Defaults to True.
        ode_solver (str, optional): ODE solver for sampling. Defaults to "dopri5_zuko".
        ode_steps (int, optional): Number of steps for ODE solver. Defaults to 100.


    Returns:
        np.array: sampled data of shape (num_jet_samples, num_particles, num_features) with features (eta, phi, pt)
        float: generation time
    """
    if verbose:
        print(f"Generating data ({num_jet_samples} samples).")
    particle_data_sampled = []
    start_time = 0

    for i in tqdm(range(num_jet_samples // batch_size), disable=not verbose):
        if cond is not None:
            cond_batch = cond[i * batch_size : (i + 1) * batch_size]
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
        remaining_samples = num_jet_samples - (num_jet_samples // batch_size * batch_size)
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

    particle_data_sampled = np.concatenate([particle_data_sampled, jet_samples_batch])
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
        self, x: torch.Tensor, mask: torch.Tensor = None, cond: torch.Tensor = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x[..., 0]).unsqueeze(-1)

        if len(x.shape) == 3:
            # for set data
            t = torch.rand_like(torch.ones(x.shape[0]))
            t = t.unsqueeze(-1).repeat_interleave(x.shape[1], dim=1).unsqueeze(-1)
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
            features, features, dim_t=2 * freqs, dim_cond=1, activation=activation
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
            node = NeuralODE(wrapped_cnf, solver="midpoint", sensitivity="adjoint")
            t_span = torch.linspace(1.0, 0.0, ode_steps)
            traj = node.trajectory(z, t_span)
            return traj[-1]
        else:
            raise NotImplementedError(f"Solver {ode_solver} not implemented")

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("log_prob not implemented")
