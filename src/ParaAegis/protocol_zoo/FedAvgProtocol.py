from ..client_zoo import BaseClient
import ray

class FedAvgProtocol:

    def __init__(self, server, clients):
        self.server = server
        self.clients = clients