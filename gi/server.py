from __future__ import annotations

import os
import sys

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

import gi
import lab as B
import lab.torch
from gi.client import Client
from gi.gibnn import GIBNN
from gi.distributions import NaturalNormal

from varz import Vars
import logging
import logging.config

logger = logging.getLogger()

class Server:
    """
    Server class contains:
    - clients
    - most recent posterior
        - i.e. current zs/ts/ps
    - previous posterior
        - i.e. prev zs/ts/ps
    - list of deltas (change in approx likelihood terms)

    client local update: maximize ELBO = ELL - KL(q, cavity)
    cavity = old q except of own contribution
    cavity => call sample_posterior with previous iters zs/ts/ps and omit own element
    """
    def __init__(
        self, config, vs: Vars,
        model: GIBNN, clients: dict[str, Client], ps: dict[str, NaturalNormal], init_q: dict[str, NaturalNormal]=None, data=None, val_data=None
    ):
        self.config = config
        self.clients: dict[str, Client] = clients
        
        self.model: GIBNN = model   # Global probabilistic model.
        self.vs: Vars = vs      # Variable container

        # Posterior
        self.ps = ps            # Global prior p(θ).
        self.q = None           # Global posterior q(θ).
        self.init_q = init_q    # Initial q(θ) for first client update.

        self.current_client = 0

        # Union of clients data, if not provided
        self.data = {
                k: B.concat([client.data[k] for client in self.clients], axis=0)
                for k in self.clients[0].data.keys()
            } if data is None else data

    def pvi(self):
        epochs = self.config.pvi_epochs
        for i in range(epochs):
            
            # Get clients to optimize
            b: list[Client] = self.select_clients()

            # Optimize client-local ts
            delta_ts: list[NaturalNormal] = [] # to save changes in t
            for client in b:
                # Communicate old posterior
                key, _delta_t = client.update_q(key, self.vs, self.ps, self.q)          
                delta_ts.append(_delta_t)
            
            # Compute new posterior
            for _dt in delta_ts:
                self.q = self.q * _dt

    def select_clients(self):
        """ 
        :returns: list of clients according to some schedule to optimize.
        
        Currently just iterates over the clients in creation order.
        """
        assert len(self.clients) != 0
        _client_name = list(self.clients.keys())[self.current_client]
        _client = self.clients[_client_name]
        self.current_client = (self.current_client + 1) % len(self.clients)
        return [_client]

        