from copy import copy
from typing import Callable, Optional
import lab as B
import torch
import logging

from experiments.kl import KL, compute_kl
from gi.models.bnn import BaseBNN


logger = logging.getLogger()


class GIBNN(BaseBNN):
    def __init__(self, nonlinearity, bias: bool, kl: KL):
        super().__init__(nonlinearity, bias, kl)

    def process_z(self, zs: dict, S: B.Int):
        """Shape zs into appropriate form and return separate dictionary. (Dicts are pass-by-reference.)
        Args:
            zs (dict): inducing points to shape
            S (B.Int): number of (batch) samples to create

        Returns:
            dict: shaped inducing points, ready to be propagated
        """
        _zs: dict = {}
        for client_name, client_z in zs.items():
            assert len(client_z.shape) == 2  # [M, Din]

            client_z = B.to_active_device(client_z)

            # Add bias vector: [M x Din] to [M x Din+bias]
            if self.bias:
                _bias = B.ones(*client_z.shape[:-1], 1)
                _cz = B.concat(client_z, _bias, axis=-1)
            else:
                _cz = client_z

            # z is [M, D]. Change to [S, M, D]]
            _zs[client_name] = B.tile(_cz, S, 1, 1)  # only tile intermediate results

        return _zs

    def propagate_z(self, zs: dict, w: B.Numeric, nonlinearity: bool):
        """Propagate inducing points through the BNN

        Args:
            zs (dict): inducing points
            w (B.Numeric): sampled weights
            nonlinearity (bool, optional): Apply nonlinearity to the outputs. Defaults to True.
        """
        for client_name, client_z in zs.items():
            # Forward the inducing inputs
            client_z = B.mm(client_z, w, tr_b=True)  # [S x M x Dout]

            if nonlinearity:  # non-final layer
                client_z = B.to_active_device(self.nonlinearity(client_z))

                # Add bias vector to any intermediate outputs
                if self.bias:
                    _bias = B.ones(*client_z.shape[:-1], 1)
                    client_z = B.concat(client_z, _bias, axis=-1)

            # Always store in _zs
            zs[client_name] = client_z

    def sample_posterior(
        self,
        key: B.RandomState,
        ps: dict,
        ts: dict,
        zs: dict,
        S: B.Int,
        cavity_client: Optional[str] = None,
    ):
        """Samples weights from the posterior distribution q.

        Args:
            key (B.RandomState): _description_
            ps (dict): priors. dict<k=layer_name, v=_p>
            ts (dict): pseudo-likelihoods. dict<k='layer x', v=dict<k='client x', v=t>>
            zs (dict): client-local inducing intputs. dict<k=client_name, v=inducing inputs>
            S (B.Int): number of samples to draw (and thus propagate)
            cavity_client (str): If provided, client to exclude for cavity distribution.

            M inducing points,
            D input space dimensionality
        Returns:
            _type_: _description_
        """

        # Shape inducing inputs for propagation; separate dict to modify
        _zs = self.process_z(zs, S)

        # Construct posterior and prior, sample, propagate.
        for i, (layer_name, p) in enumerate(ps.items()):

            # Init posterior to prior
            q = p
            p_ = p

            # Compute new posterior by multiplying client factors
            for client_name, t in ts[layer_name].items():
                _t = t(_zs[client_name])
                q *= _t  # propagate prev layer's inducing outputs

                if cavity_client and client_name != cavity_client:
                    p_ *= _t

            # Sample q, compute KL wrt (cavity) prior, and store drawn weights.
            key, w = self._sample_posterior(key, q, p_, layer_name)

            # Propagate client-local inducing inputs <z> and store prev layer outputs in _zs
            if i < len(ps.keys()) - 1:
                self.propagate_z(_zs, w, nonlinearity=True)
            else:
                self.propagate_z(_zs, w, nonlinearity=False)

        return key, self._cache

    @property
    def S(self):
        """Returns cached number of weight samples"""
        return self._cache["layer0"]["w"].shape[0]

    def compute_ell(self, out, y):
        raise NotImplementedError

    def compute_error(self, out, y):
        raise NotImplementedError


class GIBNN_Regression(GIBNN):
    def __init__(self, nonlinearity, bias: bool, kl: KL, likelihood: Callable):
        super().__init__(nonlinearity, bias, kl)
        self.likelihood = likelihood
        self.error_metric = "error"

    def compute_ell(self, out, y):
        if y.device != out.device:
            y = y.to(out.device)
        return self.likelihood(out).log_prob(y).sum(-1).mean(-1)

    def compute_error(self, out, y):
        if y.device != out.device:
            y = y.to(out.device)
        error = (y - out.mean(0)).detach().clone()  # error of mean prediction
        rmse = B.sqrt(B.mean(error**2))
        return rmse

    def performance_metrics(self, loader):
        if B.ActiveDevice.active_name and B.ActiveDevice.active_name.__contains__("cuda") and (not loader.dataset[0][0].device.type == "cuda"):
            loader.pin_memory = True

        rmses = 0.0
        mlls = 0.0
        for batch_idx, (x_mb, y_mb) in enumerate(loader):
            y_pred = self(x_mb)
            mll = self.compute_ell(y_pred, y_mb)  # [S]

            rmses += self.compute_error(y_pred, y_mb)
            mlls = ((mlls * batch_idx) + mll.mean()) / (batch_idx + 1)

        rmse = rmses / len(loader)
        return {"mll": mlls, self.error_metric: rmse}


class GIBNN_Classification(GIBNN):
    def __init__(self, nonlinearity, bias: bool, kl: KL):
        super().__init__(nonlinearity, bias, kl)
        self.error_metric = "acc"

    def compute_ell(self, out, y):
        _y = B.tile(B.to_active_device(y), out.shape[0], 1, 1)  # reshape y into [S x N x Dout]
        assert _y.shape == out.shape, "These need to be the same shape."
        return torch.distributions.Categorical(logits=out).log_prob(torch.argmax(_y, dim=-1)).mean(-1)

    def compute_error(self, out, y):
        # out: [S x N x Dout]; y [N x Dout]

        output = out.log_softmax(-1).logsumexp(0) - B.log(out.shape[0])
        pred = output.argmax(dim=-1).cpu()
        accuracy = pred.eq(torch.argmax(y, dim=1).view_as(pred)).float().mean()

        del y
        del pred
        return 1 - accuracy

    def performance_metrics(self, loader):
        if B.ActiveDevice.active_name and B.ActiveDevice.active_name.__contains__("cuda") and (not loader.dataset[0][0].device.type == "cuda"):
            loader.pin_memory = True
        correct = 0
        mlls = 0.0
        for batch_idx, (x_mb, y_mb) in enumerate(loader):
            y_pred = self(x_mb)  # one-hot encoded
            mll = self.compute_ell(y_pred, y_mb)  # [S]
            error = self.compute_error(y_pred, y_mb)
            correct += (1 - error) * y_mb.shape[0]

            mlls = ((mlls * batch_idx) + mll.mean()) / (batch_idx + 1)

        N = loader.dataset.tensors[1].shape[0]
        acc = correct / N
        return {"mll": mlls, self.error_metric: acc}
