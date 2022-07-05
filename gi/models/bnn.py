import lab as B
import torch
import logging


class BaseBNN:
    def __init__(self) -> None:
        self._cache = {}

    def propagate(self, x):
        """Propagates input through BNN using S cached posterior weight samples.

        :param x: input data

        Returns: output values from propagating x through BNN
        """
        if self._cache is None:
            return None

        if len(x.shape) == 2:
            x = B.tile(x, self.S, 1, 1)
            if self.bias:
                _bias = B.ones(*x.shape[:-1], 1)
                x = B.concat(x, _bias, axis=-1)

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
