from __future__ import annotations
import lab as B
import torch
import logging

from experiments.kl import KL, compute_kl
from gi.distributions import NaturalNormal, NaturalNormalFactor
from gi.models.bnn import BaseBNN


logger = logging.getLogger()


class MFVI(BaseBNN):
    def __init__(self, nonlinearity, bias: bool, kl: KL) -> None:
        super().__init__(nonlinearity, bias, kl)

    def process_x(self, x, S: B.Int):
        """Shape xs into appropriate form and return separate dictionary. (Dicts are pass-by-reference.)
        Args:
            xs (dict): inputs to shape
            S (B.Int): number of (batch) samples to create

        Returns:
            dict: shaped inputs, ready to be propagated
        """
        assert len(x.shape) == 2

        # Add bias vector: [N x Din] to [N x Din+bias]
        if self.bias:
            _bias = B.ones(*x.shape[:-1], 1)
            _cx = B.concat(x, _bias, axis=-1)
        else:
            _cx = x

        # x is [N, D]. Change to [S, N, D]]
        _x: B.Numeric = B.tile(_cx, S, 1, 1)  # only tile intermediate results

        return _x

    def sample_posterior(self, key, qs: dict[str, dict[str, NaturalNormalFactor]], ps: dict[str, NaturalNormal]):

        # Construct posterior, sample, and propagate
        for i, (layer_name, p) in enumerate(ps.items()):
            # Init posterior
            q: NaturalNormal = p

            # Build posterior. layer_client_q is NaturalNormal
            # should make NNFactor => no sampling function because the factor can have negative precision (i.e. not be a distribution)
            for (client_name, layer_client_q) in qs[layer_name].items():

                q *= layer_client_q

            # constrain q to have positive precision

            key, _ = self._sample_posterior(key, q, p, layer_name)

        return key, self.cache


class MFVI_Regression(MFVI):
    def __init__(self, nonlinearity, bias: bool, kl: KL) -> None:
        super().__init__(nonlinearity, bias, kl)

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
