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
        self.idx = 0

    def __next__(self):
        client = self.clients[self.idx]
        self.idx = (self.idx + 1) % len(self.clients)
        return [client]
