# wrapping the density estimator in a sklearn-like API
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import flows as fnn

from os import makedirs
from os.path import join
from tqdm import tqdm


class ConditionalNormalizingFlow:
    """Conditional normalizing flow based on torch but wrapped such that it
    mimicks the scikit-learn API, using numpy arrays as inputs and outputs.
    """
    def __init__(self, save_path=None,
                 model_type="MAF", transform="Affine", optimizer="Adam",
                 num_inputs=4, num_cond_inputs=1, num_blocks=15,
                 num_hidden=128, activation_function="relu",
                 pre_exp_tanh=False, batch_norm=True, batch_norm_momentum=1,
                 lr=0.0001, weight_decay=0.000001, no_gpu=False):

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

    def fit(self, m_train, x_train, m_val, x_val,
            batch_size=256, epochs=100, verbose=False):

        makedirs(self.de_model_path, exist_ok=True)

        # build data loader out of numpy arrays
        train_loader = numpy_to_torch_loader(m_train, x_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             device=self.device)
        val_loader = numpy_to_torch_loader(m_val, x_val,
                                           batch_size=batch_size,
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
            np.save(join(self.save_path, "DE_train_losses.npy"), train_losses)
            np.save(join(self.save_path, "DE_val_losses.npy"), val_losses)

        # training loop
        for epoch in range(epochs):
            print('\nEpoch: {}'.format(epoch))

            train_loss = train_epoch(self.model, self.optimizer, train_loader,
                                     device=torch.device("cpu"),
                                     verbose=verbose)[0]
            val_loss = compute_loss_over_batches(self.model, val_loader,
                                                 device=self.device)[0]

            print("train_loss = ", train_loss)
            print("val_loss = ", val_loss)
            train_losses = np.concatenate(
                (train_losses, np.array([train_loss])))
            val_losses = np.concatenate(
                (val_losses, np.array([val_loss])))

            if self.save_path is not None:
                np.save(join(self.save_path, "DE_train_losses.npy"),
                        train_losses)
                np.save(join(self.save_path, "DE_val_losses.npy"),
                        val_losses)
                self._save_model(join(self.de_model_path,
                                      f"DE_epoch_{epoch}.par"))

        self.model.eval()

    def sample(self, m):
        m_torch = torch.from_numpy(
            m).reshape((-1, 1)).type(torch.FloatTensor).to(self.device)
        x_torch = self.model.sample(num_samples=len(m_torch),
                                    cond_inputs=m_torch)
        return x_torch.cpu().detach().numpy()

    def log_probs(self, m, x):
        m_torch = torch.from_numpy(
            m).reshape((-1, 1)).type(torch.FloatTensor).to(self.device)
        x_torch = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
        log_prob = self.model.log_probs(x_torch, m_torch)
        return log_prob.detach().cpu().numpy().flatten()

    def load_best_model(self):
        if self.save_path is None:
            raise ValueError("save_path is None, cannot load best model")
        val_losses = np.load(join(self.save_path, "DE_val_losses.npy"))
        best_epoch = np.argmin(val_losses)
        self._load_model(
            join(self.de_model_path, f"DE_epoch_{best_epoch}.par"))
        self.model.eval()

    def _load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path,
                                              map_location=self.device))

    def _save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)


# utility functions

def numpy_to_torch_loader(m, x, batch_size=256, shuffle=True,
                          device=torch.device("cpu")):
    m_torch = torch.from_numpy(
        m).reshape((-1, 1)).type(torch.FloatTensor).to(device)
    x_torch = torch.from_numpy(x).type(torch.FloatTensor).to(device)
    dataset = torch.utils.data.TensorDataset(x_torch, m_torch)
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
    # Does one epoch of ANODE model training.

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
