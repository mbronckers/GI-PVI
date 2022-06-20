from __future__ import annotations

from typing import Optional
from gi.distributions import NormalPseudoObservation, NaturalNormal
from gi.likelihoods import NormalLikelihood
import lab as B
import lab.torch
import torch 

from gi.gibnn import GIBNN
from itertools import chain
import logging
import logging.config
from varz import Vars

logger = logging.getLogger()

class Client:
    """
    Client class that contains: client-local data, the parameters of its factor, and a function how to build the factor.

    Each client in GI-PVI has their own set of inducing points.

    :param data: {X, Y}
    :param name: Client name. Optional
    :param z: Global inducing points. (we have client-local inducing points)
    :param t: Likelihood factors per layer. Dict<k=layer, v=NormalPseudoObservation()>
    :param yz: Pseudo (inducing) observations (outputs)
    :param nz: Pseudo noise
    """
    def __init__(self, config, name: Optional[str], data: dict[str, B.Numeric], z, t: dict[str, NormalPseudoObservation], model: GIBNN):
        self.name = name if name else None
        self.data: dict[str, B.Numeric] = data
        self.z = z
        self.t = t
        
        # Shared access to nonlinearity, bias
        self.model = model

        # Client local
        self.likelihood: NormalLikelihood = None
        self.config = config
        self._parameters: list[str] = None # list of parameter names to take out of vs
        self._cache = {}

    @property
    def x(self):
        return self.data['x']

    @property
    def y(self):
        return self.data['y']

    def parameters(self, vs: Vars):
        if self._parameters == None:
            _p = list(chain.from_iterable((f"ts.{self.name}_{layer_name}_yz", f"ts.{self.name}_{layer_name}_nz") for layer_name, _t in self.t.items()))
            _p.append(f"zs.{self.name}_z")
            self._parameters = _p
        
        # Need to get latent representation to have non-leaf tensors for optimizer
        return vs.get_latent_vars(*(self._parameters))

    
    def update_nz(self, vs):
        """ Update likelihood factors' precision based on the current state of vs

        Args:
            vs: optimizable variable container
        """
        
        for i, layer_name in enumerate(self.t.keys()):
            var = vs[f"ts.{self.name}_{layer_name}_nz"]
            self.t[layer_name].nz = var

    def get_final_yz(self):
        return self.t[list(self.t.keys())[-1]].yz # final layer yz

    def transform_z(self, bias, samples):
        """ Add bias and tile <samples> of z to get input form"""
        # Add bias vector and tile for number of training samples
        if bias:
            _bias = B.ones(*self.z.shape[:-1], 1)
            _z = B.concat(self.z, _bias, axis=-1)
        
        # z is [M, D]. Change to [S, M, D]]
        _cz = B.tile(_z, samples, 1, 1) 

        return _cz

    def sample_posterior(self, key: B.RandomState,
                        q_old: dict[str, NaturalNormal]):
        """
        :returns: key
        :returns: _cz: B.Numeric = BNN output
        """
        # Get inducing points in appropriate shape and add bias
        _cz = self.transform_z(self.config.bias, self.config.S)

        for i, (layer_name, q_prev) in enumerate(q_old.items()):
            
            # Compute cavity, new client-local posterior
            _t = self.t[layer_name](_cz)
            q_cav = q_prev / _t     # TODO: q_cav.prec of layer1 is not PD
            q = q_prev * _t         # compute client-local posterior

            # Sample weights from posterior distribution q. q already has S passed via _zs
            key, w = q.sample(key) # w is [S, Dout, Din] of layer i.
            w = w[..., 0] # [S, Dout, Din]

            # Compute KL with cavity distribution
            kl_cav = q.kl(q_cav)
            kl_cav = B.sum(kl_cav, -1)  # Sum across output dimensions. [S]

            self._cache[layer_name] = {"w": w, "kl": kl_cav, "t": _t}

            # Propagate client-local inducing inputs <z>
            _cz = B.mm(_cz, w, tr_b=True)         # propagate z. [S x M x Dout]
            if i < len(q_old.keys()) - 1:  # non-final layer
                _cz = self.model.nonlinearity(_cz)      # forward and updating the inducing inputs
            
                # Add bias vector to any intermediate outputs
                if self.model.bias:
                    _bias = B.ones(*_cz.shape[:-1], 1)
                    _cz = B.concat(_cz, _bias, axis=-1)

        return key, _cz

    def update_q(self, key, vs: vars, 
                ps: dict[str, NaturalNormal], 
                q_old: dict[str, NaturalNormal]):
        """
        Client-local approximator posterior optimization
        
        :param vs: variable state; should only contain client params & be hard (separate copy)!
        
        :param ps: priors. dict<k=layer_name, v=_p>
        :param q_old: previous posterior. dict<k=layer_name, v=_q>
        """
        # If previous posterior not provided, set to prior
        q_old = q_old if q_old else ps          
    
        # batch_size = self.config.batch_size
        N = len(self.x)
        batch_size = N
        
        # Get ONLY this client's params, set single lr for all
        optim = getattr(torch.optim, self.config.optimizer)(self.parameters(vs), **self.config.optimizer_params)

        # Run update
        epochs = self.config.client_epochs
        for i in range(epochs):
            # Construct i-th minibatch {x, y} training data
            inds = (B.range(batch_size) + batch_size*i) % len(self.x)
            x_mb = B.take(self.x, inds)
            y_mb = B.take(self.y, inds)

            # Compute new local approximate posterior
            y_pred = self.sample_posterior(key, q_old)

            # Compute total cavity KL
            kl = 0.
            t_old = {}
            for layer_name, layer_cache in self._cache.items():
                kl += layer_cache["kl"]
                t_old[layer_name] = layer_cache['t']

            # Compute the expected log-likelihood.
            exp_ll = self.likelihood(y_pred).logpdf(self.y)
            exp_ll = exp_ll.mean(0).sum()       # take mean across inference 

            kl = kl.mean()                         # across inference samples
            error = y_mb-y_pred.mean(0)               # error of mean prediction
            rmse = B.sqrt(B.mean(error**2))
            
            # Mini-batching estimator of ELBO (N / batch_size)
            elbo = ((N / len(x_mb)) * exp_ll) - kl

            # Logging 
            logger.info(f"{self.name} [epoch {i}/{epochs}] - elbo: {round(elbo.item(), 0):13.1f}, ll: {round(exp_ll.item(), 0):13.1f}, kl: {round(kl.item(), 1):8.1f}, error: {round(rmse.item(), 3):3}, var: {round(y_pred.var().item(), 3):3}")

            loss = -elbo
            loss.backward()
            optim.step()
            
            # Rebuild necessary parameters
            self.update_nz(vs)
            # If using separate likelihood for each client, need to rebuild as well (see optimization.py)
            # _idx = vs.name_to_index[f"{self.name}_output_var"] 
            # self.likelihood.var = vs.transforms[_idx](vs.get_vars()[_idx])
            
            # Compute the change in approx likelihood factors
            delta_t = {}
            _cz = self.transform_z(self.config.bias, self.config.S)
            for layer_name, layer_t_old in t_old.items():
                layer_t_new = self.t[layer_name](_cz)
                delta_t[layer_name] = layer_t_new / layer_t_old

            # Remove accumulated grads
            optim.zero_grad() 

        # Communicate back the change in approximate likelihood parameters
        return key, delta_t

def build_z(key: B.RandomState, M: B.Int, x, y, random: bool=False):
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
        key, z = B.randn(key, B.default_dtype, M, *x.shape[1:]) # [M x input_dim]
        key, yz = B.randn(key, B.default_dtype, M, *y.shape[1:]) # [M x output_dim]
        return key, z, yz

    if M <= len(x):
        # Select random subset of size M of training points x
        key, perm = B.randperm(key, B.default_dtype, len(x))
        idx = perm[:M]
        z, yz = x[idx], y[idx]
    else:
        z, yz = x, y
        key, z_ = B.randn(key, B.default_dtype, M - len(x), *x.shape[1:]) # Generate z_, yz_ 
        key, yz_ = B.randn(key, B.default_dtype, M - len(x), *y.shape[1:])
        z = B.concat(z, z_)
        yz = B.concat(yz, yz_)
        
    return key, z, yz

def build_ts(key, M, yz, *dims: B.Int, nz_init: float):
    """
    Builds likelihood factors per layer for one client
    
    For the final layer, the pseudo observations are init to the passed <yz> (usually, the training data output y)
    For non-final layers, the pseudo obersvations <_yz> ~ N(0, 1)

    :return ts: Dictionary of likelihood factors for each layer.
    :rtype: dict<k=layer_name, v=NormalPseudoObservation>
    """
    ts = {}
    num_layers = len(dims) - 1
    for i in range(num_layers):   
        if i < num_layers - 1: 
            _nz = B.ones(dims[i + 1], M) * nz_init # [Dout x M]
            key, _yz = B.randn(key, B.default_dtype, M, dims[i + 1]) # [M x Dout]
            t = NormalPseudoObservation(_yz, _nz)
        else: 
            # Last layer precision gets initialized to 1 
            _nz = B.ones(dims[i + 1], M) * 1        # [Dout x M]
            t = NormalPseudoObservation(yz, _nz) # final layer
            
        ts[f"layer{i}"] = t
        
    return ts