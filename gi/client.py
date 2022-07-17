from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain

from typing import Optional, Union

import torch
from gi.distributions import MeanFieldFactor, NormalPseudoObservation
import lab as B
import lab.torch
from varz import Vars

from gi.likelihoods import NormalLikelihood


class Client:
    def __init__(self, name: Optional[str], x, y):
        self.name = name if name else None
        self.x = x
        self.y = y

        # Add optimizable client variables to vs
        self._vs: Vars = Vars(B.default_dtype)

    @property
    def vs(self):
        return self._vs

    def get_params(self, names: Optional[Union[str, list]] = None) -> list:
        """Gets the appropriate client's variables from the client-local variable manager"""
        if names != None:
            return self.vs.get_latent_vars(names)
        else:
            return self.vs.get_latent_vars()

    def __repr__(self) -> str:
        return f"{self.name}"


class GI_Client(Client):
    """
    Client class that contains: client-local data, the parameters of its factor, and container to store the optimizable parameters

    Each client in GI-PVI has their own set of inducing points.

    :param data: {X, Y}
    :param name: Client name. Optional
    :param z: Global inducing points. (we have client-local inducing points)
    :param t: Likelihood factors per layer. Dict<k=layer, v=NormalPseudoObservation()>
    :param yz: Pseudo (inducing) observations (outputs)

    :random_z: If true, we entirely init z to random samples from N(0,1)
    :param nz_inits: Pseudo observation precision's initial values.
    :param linspace_yz: If true, we use a linear space mapping [-1, 1] for the pseudo observations.
    """

    def __init__(self, key: B.RandomState, name: Optional[str], x, y, M, *dims, random_z, nz_inits, linspace_yz):
        super().__init__(name, x, y)

        # Build inducing points
        key, z, yz = self.build_z(key, M, x, y, random=random_z)
        self.z: B.Numeric = z

        # Build approximate likelihood factors
        self.key, t = self.build_ts(key, M, yz, *dims, nz_inits=nz_inits, linspace_yz=linspace_yz)
        self.t: dict[str, NormalPseudoObservation] = t

        # Add optimizable client variables to vs
        self.vs.unbounded(self.z, name=f"zs.{self.name}_z")

        for layer_name, _t in self.t.items():
            self.vs.unbounded(_t.yz, name=f"ts.{self.name}_{layer_name}_yz")
            self.vs.positive(_t.nz, name=f"ts.{self.name}_{layer_name}_nz")

        self.vs.requires_grad(True, *self.vs.names)

        # Classification related issue
        if self.z.requires_grad != True:
            self.z = self.vs[f"zs.{self.name}_z"]
            self.t["layer2"].yz = self.vs[f"ts.{self.name}_layer2_yz"]

        self.update_nz()

    def build_z(self, key: B.RandomState, M: B.Int, x, y, random: bool = False):
        """
        Build M inducing points from data (x, y).
        - If M < len(x), select a random M-sized subset from x
        - If M > len(x), init len(x) points to x, then randomly sample from N(0,1)

        :param zs: inducing inputs
        :param yz: pseudo (inducing) outputs for final layer
        :param random: if true, we entirely init z to random samples from N(0,1)

        :returns: key, z, y
        """
        if random:
            key, z = B.randn(key, B.default_dtype, M, *x.shape[1:])  # [M x input_dim]
            key, yz = B.randn(key, B.default_dtype, M, *y.shape[1:])  # [M x output_dim]
            return key, z, yz

        if M == len(x):
            # Initialize inducing points to data
            z = x.clone()
            yz = y.clone()
        elif M < len(x):
            # Select random subset of size M of training points x
            key, perm = B.randperm(key, torch.float64, len(x))
            idx = perm[:M]
            z, yz = x[idx].clone(), y[idx].clone()
        else:
            z, yz = x.clone(), y.clone()
            key, z_ = B.randn(key, B.default_dtype, M - len(x), *x.shape[1:])  # Generate z_, yz_
            key, yz_ = B.randn(key, B.default_dtype, M - len(x), *y.shape[1:])
            z = B.concat(z, z_)
            yz = B.concat(yz, yz_)

        return key, z, yz

    def build_ts(self, key, M, yz, *dims: B.Int, nz_inits: list[float], linspace_yz: bool = False):
        """
        Builds likelihood factors per layer for one client

        For the final layer, the pseudo observations are init to the passed <yz> (usually, the training data output y)
        For non-final layers, the pseudo obersvations <_yz> ~ N(0, 1)

        :return ts: Dictionary of likelihood factors for each layer.
        :rtype: dict<k=layer_name, v=NormalPseudoObservation>
        """
        ts = {}
        num_layers = len(dims) - 1
        assert len(nz_inits) == num_layers
        for i, nz_init in enumerate(nz_inits):
            _nz = B.ones(dims[i + 1], M) * nz_init  # [Dout x M]
            if i < num_layers - 1:
                # Initialization to M linspace vectors: [M x Dout]
                if linspace_yz:
                    _yz, _ = torch.meshgrid(B.linspace(-1, 1, dims[i + 1]), B.ones(M))
                    _yz = _yz.transpose(-1, -2)
                    t = NormalPseudoObservation(_yz.detach().clone(), _nz)

                else:
                    key, _yz = B.randn(key, B.default_dtype, M, dims[i + 1])  # [M x Dout]
                    t = NormalPseudoObservation(_yz, _nz)

            else:
                if yz.dtype != B.default_dtype:
                    yz = yz.to(B.default_dtype)

                t = NormalPseudoObservation(yz, _nz)  # final layer

            ts[f"layer{i}"] = t

        return key, ts

    def update_nz(self):
        """Update likelihood factors' precision based on the current state of vs

        Args:
            vs: optimizable variable container
        """
        for i, layer_name in enumerate(self.t.keys()):
            _prec = self.vs[f"ts.{self.name}_{layer_name}_nz"]
            self.t[layer_name].nz = _prec

    def get_final_yz(self):
        return self.t[list(self.t.keys())[-1]].yz  # final layer yz


class MFVI_Client(Client):
    def __init__(self, name: Optional[str], x, y, *dims: B.Int, prec_inits: list[float], bias: bool = True, S):
        super().__init__(name, x, y)

        self.bias = bias

        # Layer posterior factors
        self.t: dict[str, MeanFieldFactor] = self.build_t(*dims, prec_inits=prec_inits, S=S)

        for layer_name, factor in self.t.items():
            self.vs.unbounded(factor.lam, name=f"ts.{self.name}_{layer_name}_yz")
            self.vs.unbounded(factor.prec.diag, name=f"ts.{self.name}_{layer_name}_nz")

        self.vs.requires_grad(True, *self.vs.names)

    def build_t(self, *dims, prec_inits: list[float], S):
        ts = {}
        num_layers = len(dims) - 1
        assert len(prec_inits) == num_layers
        for i, prec_init in enumerate(prec_inits):
            if self.bias:
                Din = dims[i] + 1
            else:
                Din = dims[i]
            lam = B.zeros(B.default_dtype, dims[i + 1], Din, 1)  # [Dout x Din]
            _prec = B.eye(B.default_dtype, Din) * prec_init
            _prec = B.tile(_prec, dims[i + 1], 1, 1)  # [Dout x Din+bias x Din+bias], i.e. [batch x Din x Din]

            ts[f"layer{i}"] = MeanFieldFactor(lam, B.diag_extract(_prec))
        return ts
