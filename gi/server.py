from __future__ import annotations
from collections import defaultdict
import logging

import sys
import os

import torch


file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

from gi import Client
from gi.gibnn import GIBNN

from experiments.utils.colors import Color

logger = logging.getLogger()


class Server:
    def __init__(self, clients: list[Client], model: GIBNN):

        self.clients = clients

        self.model = model

        self.log = defaultdict(list)

        self.max_global_iters = None
        self.curr_global_iter = 0

    def __iter__(self):
        return self

    def update_log(self, metrics: defaultdict):
        for k, v in metrics.items():
            self.log[k].append(v)

    def evaluate_performance(self):
        """Runs evaluation of the global (shared) model on training and test data."""
        with torch.no_grad():

            # Get performance metrics.
            train_metrics = self.model.performance_metrics(self.train_loader)
            test_metrics = self.model.performance_metrics(self.test_loader)
            error_key = self.model.error_metric
            logger.info(
                "SERVER - {} - iter [{:2}/{:2}] - {}train mll: {:8.3f}, train {}: {:8.3f}, test mll: {:8.3f}, test {}: {:8.3f}{}".format(
                    self.name,
                    self.curr_iter,
                    self.max_global_iters,
                    Color.BLUE,
                    train_metrics["mll"],
                    error_key,
                    train_metrics[error_key],
                    test_metrics["mll"],
                    error_key,
                    test_metrics[error_key],
                    Color.END,
                )
            )

            # Save metrics.
            train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
            test_metrics = {"test_" + k: v for k, v in test_metrics.items()}
            metrics = {**train_metrics, **test_metrics}
            self.log["iteration"].append(self.curr_global_iter)
            for k, v in metrics.items():
                self.log[k].append(v.item())


class SynchronousServer(Server):
    def __init__(self, clients: list[Client], model: GIBNN, iters: int):
        super().__init__(clients, model)
        self.name = "synchronous"

        self.max_global_iters = iters

    def __next__(self):
        logger.info(f"SERVER - {self.name} - iter [{self.curr_global_iter+1:2}/{self.max_iters}] - optimizing {list(self.clients.keys())}")

        return list(self.clients.values())


class SequentialServer(Server):
    def __init__(self, clients: list[Client], model: GIBNN, iters: int):
        super().__init__(clients, model)
        self.name = "sequential"
        self._idx = 0

        # Want to have equal number of posterior updates for same number of iterations.
        self.max_global_iters = iters * len(self.clients)

    def current_client(self):
        return self.clients[list(self.clients.keys())[self._idx]]

    def __next__(self):
        client = self.current_client()
        self._idx = (self._idx + 1) % len(self.clients)

        logger.info(f"SERVER - {self.name} - iter [{self.curr_global_iter+1:2}/{self.max_global_iters}] - optimizing {self.clients}")

        return [client]
