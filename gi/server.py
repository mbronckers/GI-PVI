import sys
import os
file_dir = os.path.dirname(__file__)
_root_dir = os.path.abspath(os.path.join(file_dir, ".."))
sys.path.insert(0, os.path.abspath(_root_dir))

from gi import Client

class Server:
    def __init__(self, clients: list[Client]):
        self.clients = clients
        
    def __iter__(self):
        return self

class SynchronousServer(Server):
    def __next__(self):
        return self.clients
        
class SequentialServer(Server):
    def __init__(self, clients: list[Client]):
        super().__init__(clients)
        self._idx = 0

    def current_client(self):
        return self.clients[list(self.clients.keys())[self._idx]]

    def __next__(self):
        client = self.current_client()
        self._idx = (self._idx + 1) % len(self.clients)
        return [client]
