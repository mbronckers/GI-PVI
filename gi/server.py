from __future__ import annotations

import os
import sys

file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

import lab as B
import lab.torch
from client import Client
from gibnn import GIBNN

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
        self, model: GIBNN, clients: dict[str, Client], p, init_q, data=None, val_data=None
    ):

        # Shared probabilistic model.
        self.model = model

        # Global prior p(θ).
        self.p = p

        # Global posterior q(θ).
        self.q = p

        # Initial q(θ) for first client update.
        self.init_q = init_q

        # Clients.
        self.clients = clients

        # Union of clients data
        if data is None:
            self.data = {
                k: B.concat([client.data[k] for client in self.clients], axis=0)
                for k in self.clients[0].data.keys()
            }
        else:
            self.data = data

    def update_hyperparams(self):
        """
        Updates the model hyperparameters according to
        dF / dε = (μ_q - μ_0)^T dη_0 / dε + Σ_m dF_m / dε.
        """

        # Zero-grad the parameters accumulated during client optimization.

        # Compute variational free energy = ELL - KL
        # vfe = 0.

        # Ensure clients have same model as server and get expected log-likelihood.

        # Compute KL divergence(q, prior)

        # Divide VFE by N (improve stability?).

        # Compute gradients
        # vfe.backward()

        # Update parameters manually
        # for p_name, p in get_vs_state.items():
            # p.data += config.lr * p.grad

        # Pass updated parameters to client

        