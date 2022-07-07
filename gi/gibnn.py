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
                client_z = self.nonlinearity(client_z)

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
        ts_p: dict,
        zs_p: dict,
        S: B.Int,
    ):
        """_summary_

        Args:
            key (B.RandomState): _description_
            ps (dict): priors. dict<k=layer_name, v=_p>
            ts (dict): pseudo-likelihoods. dict<k='layer x', v=dict<k='client x', v=t>>
            zs (dict): client-local inducing intputs. dict<k=client_name, v=inducing inputs>
            S (B.Int): number of samples to draw (and thus propagate)
            ts_p (dict, optional): pseudo-likelihoods used to define the prior.
            zs_p (dict, optional): client-local inducing inputs used to define the prior.

            M inducing points,
            D input space dimensionality
        Returns:
            _type_: _description_
        """

        # Shape inducing inputs for propagation; separate dict to modify
        _zs = self.process_z(zs, S)
        if zs_p:
            _zs_p = self.process_z(zs_p, S)

        # Construct posterior and prior, sample, propagate.
        for i, (layer_name, p) in enumerate(ps.items()):

            # Init posterior to prior
            q = p
            p_ = p

            # Compute new posterior by multiplying client factors
            for client_name, t in ts[layer_name].items():
                q *= t(_zs[client_name])  # propagate prev layer's inducing outputs

            # Compute prior distribution by multiplying factors
            if ts_p:
                for client_name, t in ts_p[layer_name].items():
                    p_ *= t(_zs_p[client_name])

            # Sample q, compute KL, and store drawn weights.
            key, w = self._sample_posterior(key, q, p, layer_name)

            # Propagate client-local inducing inputs <z> and store prev layer outputs in _zs
            if i < len(ps.keys()) - 1:
                self.propagate_z(_zs, w, nonlinearity=True)
                if zs_p:
                    self.propagate_z(_zs_p, w, nonlinearity=True)
            else:
                self.propagate_z(_zs, w, nonlinearity=False)
                if zs_p:
                    self.propagate_z(_zs_p, w, nonlinearity=False)

        return key, self._cache

    @property
    def S(self):
        """Returns cached number of weight samples"""
        return self._cache["layer0"]["w"].shape[0]
