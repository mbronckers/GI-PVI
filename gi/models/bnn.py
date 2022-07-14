import lab as B
import torch
import logging

from experiments.kl import KL, compute_kl


class BaseBNN:
    def __init__(self, nonlinearity, bias: bool, kl: KL) -> None:
        self._cache = {}
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.kl = kl

    def _sample_posterior(self, key, q, p, layer_name):
        # Sample weights from posterior distribution q. q already has S passed in its parameters.
        key, w = q.sample(key)  # w is [S, Dout, Din] of layer i.

        # Compute KL divergence between prior and posterior
        kl_qp = compute_kl(self.kl, q, p, w)

        # Sum across output dimensions.
        kl_qp = B.sum(kl_qp, -1)  # [S]

        # Get rid of last dimension.
        w = w[..., 0]  # [S, Dout, Din]

        # Save posterior w samples and KL to cache
        self._cache[layer_name] = {"w": w, "kl": kl_qp}

        return key, w

    def propagate(self, x):
        """Propagates input through BNN using S cached posterior weight samples.

        :param x: input data

        Returns: output values from propagating x through BNN
        """
        if self._cache is None:
            return None

        if len(x.shape) == 2:
            x = B.tile(B.to_active_device(x), self.S, 1, 1)
            if self.bias:
                _bias = B.ones(*x.shape[:-1], 1)
                x = B.concat(x, _bias, axis=-1)
        else:
            x = B.to_active_device(x)

        for i, (layer_name, layer_dict) in enumerate(self._cache.items()):
            x = B.mm(x, layer_dict["w"], tr_b=True)
            if i < len(self._cache.keys()) - 1:  # non-final layer
                x = self.nonlinearity(x)

                if self.bias:
                    _bias = B.ones(*x.shape[:-1], 1)
                    x = B.concat(x, _bias, axis=-1)

        return x

    def __call__(self, x):
        return self.propagate(x)

    def get_total_kl(self):
        if self.cache == None:
            return None

        kl_qp = 0.0
        for layer_dict in self.cache.values():
            kl_qp += layer_dict["kl"]

        return kl_qp

    @property
    def S(self):
        """Returns cached number of weight samples"""
        raise NotImplementedError

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache
