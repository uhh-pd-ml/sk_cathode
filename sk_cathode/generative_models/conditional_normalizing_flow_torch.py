# wrapping the density estimator in a sklearn-like API
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sk_cathode.generative_models.flows as fnn

from os import makedirs
from os.path import join
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class ConditionalNormalizingFlow:
    """Conditional normalizing flow based on torch but wrapped such that it
    mimicks the scikit-learn API, using numpy arrays as inputs and outputs.
    """
    def __init__(self, save_path=None, load=False,
                 model_type="MAF", transform="Affine", optimizer="Adam",
                 num_inputs=4, num_cond_inputs=1, num_blocks=15,
                 num_hidden=128, activation_function="relu",
                 pre_exp_tanh=False, batch_norm=True, batch_norm_momentum=1,
                 lr=0.0001, weight_decay=0.000001, early_stopping=False,
                 patience=10, no_gpu=False, val_split=0.2,
                 batch_size=256, epochs=100, verbose=False):

        if model_type != "MAF":
            raise NotImplementedError
        if transform != "Affine":
            raise NotImplementedError
        if optimizer != "Adam":
            raise NotImplementedError

        self.save_path = save_path
        self.de_model_path = join(save_path, "DE_models/")
    
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   and not no_gpu else "cpu")
        self.early_stopping = early_stopping
        self.patience = patience
        self.val_split = val_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        modules = []
        for i in range(num_blocks):
            modules += [
                fnn.MADE(num_inputs, num_hidden,
                         num_cond_inputs,
                         act=activation_function,
                         pre_exp_tanh=pre_exp_tanh),
                ]
            if batch_norm:
                modules += [fnn.BatchNormFlow(num_inputs,
                                              momentum=batch_norm_momentum)]
            modules += [fnn.Reverse(num_inputs)]

        self.model = fnn.FlowSequential(*modules)

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
        # Workaround bug in flow.py
        self.model.num_inputs = num_inputs

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

        self.load_best_model()

    def transform(self, X, m=None):

        # m needs to be provided, but trying to mimick the sklearn API here
        if m is None:
            raise ValueError("m needs to be provided!")

        X_torch = torch.from_numpy(X).type(torch.FloatTensor).to(self.device)
        m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(self.device)
        Z = self.model(X_torch, m_torch)[0]

        return Z.cpu().detach().numpy()

    def sample(self, n_samples=1, m=None):

        # m needs to be provided, but trying to mimick the sklearn API here
        if m is None:
            raise ValueError("m needs to be provided!")

        m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(self.device)
        X_torch = self.model.sample(num_samples=n_samples,
                                    cond_inputs=m_torch)
        return X_torch.cpu().detach().numpy()

    def predict_log_proba(self, X, m=None):

        # m needs to be provided, but trying to mimick the sklearn API here
        if m is None:
            raise ValueError("m needs to be provided!")

        X_torch = torch.from_numpy(X).type(torch.FloatTensor).to(self.device)
        m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(self.device)
        log_prob = self.model.log_probs(X_torch, m_torch)
        return log_prob.detach().cpu().numpy().flatten()

    def predict_proba(self, X, m=None):
        return np.exp(self.predict_log_proba(X, m=m))

    def score_samples(self, X, m=None):
        return self.predict_log_proba(X, m=m)

    def score(self, X, m=None):
        return self.score_samples(X, m=m).sum()

    def load_best_model(self):
        val_losses = self.load_val_loss()
        best_epoch = np.argmin(val_losses) - 1  # includes pre-training loss
        self.load_epoch_model(best_epoch)
        self.model.eval()

    def load_train_loss(self):
        if self.save_path is None:
            raise ValueError("save_path is None, cannot load train loss")
        return np.load(self._train_loss_path())

    def load_val_loss(self):
        if self.save_path is None:
            raise ValueError("save_path is None, cannot load val loss")
        return np.load(self._val_loss_path())

    def load_epoch_model(self, epoch):
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
    X_torch = torch.from_numpy(X).type(torch.FloatTensor).to(device)
    m_torch = torch.from_numpy(m).type(torch.FloatTensor).to(device)
    dataset = torch.utils.data.TensorDataset(X_torch, m_torch)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
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

            loss_vals_raw = model.log_probs(data, cond_data)
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
    # Does one epoch of flow model training.

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
        loss = -model.log_probs(data, cond_data)
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

    has_batch_norm = False
    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            has_batch_norm = True
            module.momentum = 0

    if has_batch_norm:
        with torch.no_grad():
            model(data_loader.dataset.tensors[0].to(data.device),
                  data_loader.dataset.tensors[1].to(data.device).float())

        for module in model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 1

    return (np.array(train_loss_avg).flatten().mean(), )
