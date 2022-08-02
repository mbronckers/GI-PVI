from __future__ import annotations
from typing import Callable, Optional
import lab as B
import torch
import logging

from experiments.kl import KL, compute_kl
from gi.distributions import MeanField, MeanFieldFactor, NaturalNormal
from gi.models.bnn import BaseBNN


logger = logging.getLogger()


class MFVI(BaseBNN):
    def __init__(self, nonlinearity, bias: bool, kl: KL) -> None:
        super().__init__(nonlinearity, bias, kl)

    def _sample_posterior(self, key, q, p, layer_name, S: int = 1):
        # Sample weights from the posterior distribution q.
        key, w = q.sample(key, S)

        kl_qp = compute_kl(self.kl, q, p, w)
        kl_qp = B.sum(kl_qp, -1)

        w = w[..., 0]

        # Save posterior w samples and KL to cache
        self._cache[layer_name] = {"w": w, "kl": kl_qp}

        return key, w

    def sample_posterior(
        self,
        key,
        ps: dict[str, NaturalNormal],
        ts: dict[str, dict[str, MeanFieldFactor]],
        S: B.Int,
        cavity_client: Optional[str] = None,
        **kwargs,
    ):

        # Construct posterior, sample, and propagate
        for i, (layer_name, p) in enumerate(ps.items()):

            # Init posterior & cavity distribution
            q = p
            p_ = p  # cavity

            # Build posterior. layer_client_q is MeanFieldFactor
            for client_name, layer_client_q in ts[layer_name].items():
                q *= layer_client_q

                if cavity_client and client_name != cavity_client:
                    p_ *= layer_client_q

            # Constrain q to have positive precision.
            # Reduce to diagonal
            q = MeanField.from_factor(q)
            p_ = MeanField.from_factor(p_)

            key, _ = self._sample_posterior(key, q, p_, layer_name, S)

        return key, self.cache

    @property
    def S(self):
        """Returns cached number of weight samples"""
        return self._cache["layer0"]["w"].shape[0]


class MFVI_Regression(MFVI):
    def __init__(self, nonlinearity, bias: bool, kl: KL, likelihood: Callable) -> None:
        super().__init__(nonlinearity, bias, kl)
        self.likelihood = likelihood
        self.error_metric = "error"

    def compute_ell(self, out, y):
        if y.device != out.device:
            y = y.to(out.device)
        return self.likelihood(out).log_prob(y).sum(-1).mean(-1)  # [S x N x Dout] => [S]

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


class MFVI_Classification(MFVI):
    def __init__(self, nonlinearity, bias: bool, kl: KL) -> None:
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
