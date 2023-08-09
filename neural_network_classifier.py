# wrapping the classifer in a sklearn-like API
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from os import makedirs
from os.path import join
from sklearn.utils import class_weight
from tqdm import tqdm


class NeuralNetwork(nn.Module):
    def __init__(self, layers=[64, 64, 64], n_inputs=4):
        super().__init__()

        self.layers = []
        for nodes in layers:
            self.layers.append(nn.Linear(n_inputs, nodes))
            self.layers.append(nn.ReLU())
            n_inputs = nodes
        self.layers.append(nn.Linear(n_inputs, 1))
        self.layers.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model_stack(x)


class NeuralNetworkClassifier:
    """Neural network classifier based on torch but wrapped such that it
    mimicks the scikit-learn API, using numpy arrays as inputs and outputs.
    """
    def __init__(self, save_path=None, n_inputs=4, layers=[64, 64, 64],
                 lr=0.001, no_gpu=False):

        self.save_path = save_path
        self.clsf_model_path = join(save_path, "CLSF_models/")

        self.model = NeuralNetwork(layers, n_inputs=n_inputs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss = F.binary_cross_entropy
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   and not no_gpu else "cpu")

        # defaulting to eval mode, switching to train mode in fit()
        self.model.eval()

    def predict(self, x):
        with torch.no_grad():
            self.model.eval()
            x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
            prediction = self.model.forward(x).detach().cpu().numpy()
        return prediction

    def fit(self, x_train, y_train, x_val, y_val,
            batch_size=128, epochs=100, use_class_weights=True, verbose=False):

        # TODO replace verbose batch printouts by tqdm progress bar?

        makedirs(self.clsf_model_path, exist_ok=True)

        # deduce class weights for training and validation sets
        # (move outside class as in sklearn?)
        if use_class_weights:
            class_weights_train = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train)
            class_weights_train = dict(enumerate(class_weights_train))

            class_weights_val = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_val), y=y_val)
            class_weights_val = dict(enumerate(class_weights_val))
        else:
            class_weights_train = None
            class_weights_val = None

        # build data loader out of numpy arrays
        train_loader = numpy_to_torch_loader(x_train, y_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             device=self.device)
        val_loader = numpy_to_torch_loader(x_val, y_val,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           device=self.device)

        # training loop
        self.model.train()
        for epoch in range(epochs):
            print('\nEpoch: {}'.format(epoch))
            pbar = tqdm(total=len(train_loader.dataset))
            epoch_train_loss = 0.
            epoch_val_loss = 0.

            for i, batch in enumerate(train_loader):
                batch_inputs, batch_labels = batch
                batch_inputs, batch_labels = (batch_inputs.to(self.device),
                                              batch_labels.to(self.device))

                # translating class weights to sample weights
                # (legacy from allowing CWoLa sample weights directly, remove?)
                if class_weights_train is not None:
                    batch_weights = class_weight_to_sample_weight(
                        batch_labels, class_weights_train)
                    batch_weights = batch_weights.type(
                        torch.FloatTensor).to(self.device)
                else:
                    batch_weights = None

                self.optimizer.zero_grad()
                batch_outputs = self.model(batch_inputs)
                batch_loss = self.loss(batch_outputs, batch_labels,
                                       weight=batch_weights)
                batch_loss.backward()
                self.optimizer.step()
                epoch_train_loss += batch_loss.item()
                if verbose:
                    pbar.update(batch_inputs.size(0))
                    pbar.set_description(
                        "Train loss: {:.6f}".format(
                            epoch_train_loss / (i+1)))

            epoch_train_loss /= (i+1)
            if verbose:
                pbar.close()

            with torch.no_grad():
                self.model.eval()
                for i, batch in enumerate(val_loader):
                    batch_inputs, batch_labels = batch
                    batch_inputs, batch_labels = (batch_inputs.to(self.device),
                                                  batch_labels.to(self.device))
                    if class_weights_val is not None:
                        batch_weights = class_weight_to_sample_weight(
                            batch_labels, class_weights_val)
                        batch_weights = batch_weights.type(
                            torch.FloatTensor).to(self.device)
                    else:
                        batch_weights = None

                    batch_outputs = self.model(batch_inputs)
                    batch_loss = self.loss(batch_outputs, batch_labels,
                                           weight=batch_weights)
                    epoch_val_loss += batch_loss.item()
                epoch_val_loss /= (i+1)
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
                np.save(join(self.save_path, "CLSF_train_losses.npy"),
                        train_losses)
                np.save(join(self.save_path, "CLSF_val_losses.npy"),
                        val_losses)
                self._save_model(join(self.clsf_model_path,
                                      f"CLSF_epoch_{epoch}.par"))

    def load_best_model(self):
        if self.save_path is None:
            raise ValueError("save_path is None, cannot load best model")
        val_losses = np.load(join(self.save_path, "CLSF_val_losses.npy"))
        best_epoch = np.argmin(val_losses)
        self._load_model(
            join(self.clsf_model_path, f"CLSF_epoch_{best_epoch}.par"))
        self.model.eval()

    def _load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path,
                                              map_location=self.device))

    def _save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)


def numpy_to_torch_loader(x, y, batch_size=128, shuffle=True,
                          device=torch.device("cpu")):
    x_torch = torch.from_numpy(
        x).type(torch.FloatTensor).to(device)
    y_torch = torch.from_numpy(
        y).type(torch.FloatTensor).to(device).reshape(-1, 1)
    dataset = torch.utils.data.TensorDataset(x_torch, y_torch)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def class_weight_to_sample_weight(y, class_weights):
    return (torch.ones(y.shape) - y)*class_weights[0] + y*class_weights[1]
