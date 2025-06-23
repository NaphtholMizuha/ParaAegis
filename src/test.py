from torch.utils.data import DataLoader

from ParaAegis.client_zoo import FedAvgClient
from ParaAegis.server_zoo import FedAvgServer
from ParaAegis.protocol_zoo import fedavg
from ParaAegis.training import fetch_model, fetch_dataset, fetch_datasplitter
from ParaAegis.training import Trainer
from copy import deepcopy
import ray
import tenseal as ts

from ParaAegis.client_zoo.HeAvgClient import HeAvgClient
from ParaAegis.server_zoo import HeAvgServer

import logging




if __name__ == '__main__':
    # 设置全局日志级别为 ERROR，会隐藏所有库的 WARNING 和 INFO
    logging.basicConfig(level=logging.ERROR)
    MODEL_NAME = 'cnn'
    DATASET_NAME = 'cifar10'
    DATASET_PATH = '/share/datasets'
    SPLIT_NAME = 'iid'
    N_CLIENTS = 10
    DEVICE = 'cuda'
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    N_EPOCHS = 5
    N_ROUNDS = 200
    LR = 0.0001

    models = [fetch_model(MODEL_NAME).to(DEVICE) for _ in range(N_CLIENTS)]
    init_state = deepcopy(models[0].state_dict())
    print(Trainer._flat(init_state).shape)
    train_set, test_set = fetch_dataset(DATASET_PATH, DATASET_NAME)
    spliter = fetch_datasplitter(train_set, SPLIT_NAME, N_CLIENTS)
    train_subsets = spliter.split()

    trainers = [Trainer(
        model=models[i],
        init_state=init_state,
        train_set=train_subsets[i],
        test_set=test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        lr=LR,
        device=DEVICE
    ) for i in range(N_CLIENTS)]
    ray.init()

    # >>===========HE Context
    # he_ctx = ts.Context(ts.SCHEME_TYPE.CKKS,poly_modulus_degree=2**15, coeff_mod_bit_sizes=[60, 40, 40, 60])
    # he_ctx.generate_galois_keys()
    # he_ctx.global_scale = 2 ** 40
    #
    # sk_bytes= he_ctx.serialize(save_secret_key=True, save_galois_keys=True)
    # pk_bytes=he_ctx.serialize(save_galois_keys=True)
    # <<=============End
    # clients = [HeAvgClient.remote(trainer=trainer, n_epochs=N_EPOCHS, ckks_bytes=sk_bytes) for trainer in trainers]
    # server = HeAvgServer.remote(ckks_bytes=pk_bytes)
    clients = [FedAvgClient.remote(trainer=trainer, n_epochs=N_EPOCHS) for trainer in trainers]
    server = FedAvgServer.remote()

    fedavg(server, clients, N_ROUNDS)
    ray.shutdown()