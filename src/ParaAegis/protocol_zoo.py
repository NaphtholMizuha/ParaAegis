from src.ParaAegis.client_zoo import BaseClient
from src.ParaAegis.utils.msg import Msg, MsgType
import ray
from typing import List

def fedavg(server, clients, n_epoch):
    for epoch in range(n_epoch):
        # step 1: model training
        ray.get([client.run.remote() for client in clients])

        # step 2: update collection
        local_grads = ray.get([client.get.remote(MsgType.GRADIENT) for client in clients])

        # step 3: model aggregation
        global_grad = ray.get(server.aggregate.remote(local_grads))

        # step 4: model distribution
        ray.get([client.set.remote(global_grad) for client in clients])

def paraaegis(server, clients, n_epoch):
    for epoch in range(n_epoch):
        # step 1: model training
        ray.get([client.run.remote() for client in clients])

        # step 2: vote collection
        local_votes = ray.get([client.get.remote(MsgType.VOTE) for client in clients])

        # step 3: vote aggregation
        global_vote = ray.get(server.aggregate.remote(local_votes))

        # step 4: vote distribution
        ray.get([client.set.remote(global_vote) for client in clients])

        # step 5: update collection
        local_updates = ray.get([client.get.remote(MsgType.PARTIAL_ENCRYPTED_GRADIENT) for client in clients])

        # step 6: update aggregation
        global_update = ray.get(server.aggregate.remote(local_updates))

        # step 7: update distribution
        ray.get([client.set.remote(global_update) for client in clients])