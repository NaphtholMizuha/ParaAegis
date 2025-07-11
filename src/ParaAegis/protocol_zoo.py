from .utils.msg import Msg, MsgType
from .utils import timer
import ray
from typing import List
import polars as pl

def fedavg(server, clients, n_rounds):
    results = {
        'acc': [],
        'loss': [],
        'train_time': [],
        'upload_time': [],
        'aggregate_time': [],
        'download_time': []
    }
    t_train, t_upload, t_aggregate, t_download = 0, 0, 0, 0
    for r in range(n_rounds):
        
        with timer() as t:
            # step 1: model training
            ray.get([client.run.remote() for client in clients])
            t_train += t()

        with timer() as t:
            # step 2: update collection
            local_grads = ray.get([client.get.remote(MsgType.GRADIENT) for client in clients])
            t_upload += t()

        with timer() as t:
            # step 3: model aggregation
            global_grad = ray.get(server.aggregate.remote(local_grads))
            t_aggregate += t()

        with timer() as t:
            # step 4: model distribution
            ray.get([client.set.remote(global_grad) for client in clients])
            t_download += t()

        # step 5: model evaluation
        loss, acc = ray.get(clients[0].test.remote())

        print(f'Round {r+1}/{n_rounds} loss: {loss:.2f}, acc: {acc * 100:.2f}%')
        print(f'Train time: {t_train:.2f}s, Upload time: {t_upload:.2f}s, Aggregate time: {t_aggregate:.2f}s, Download time: {t_download:.2f}s')
        results['acc'].append(acc)
        results['loss'].append(loss)
        results['train_time'].append(t_train)
        results['upload_time'].append(t_upload)
        results['aggregate_time'].append(t_aggregate)
        results['download_time'].append(t_download)
        
    return pl.DataFrame(results)

def paraaegis(server, clients, n_rounds):
    for r in range(n_rounds):
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
        
        result = ray.get(clients[0].test.remote())

        print(f'Round {r+1}/{n_rounds} loss: {result[0]:.2f}, acc: {result[1] * 100:.2f}%')
        # swanlab.log({
        #     'acc': result[1],
        #     'loss': result[0],
        # })