from torch.utils.data import DataLoader

from ParaAegis.client_zoo import FedAvgClient, DpAvgClient, ParaAegisClient
from ParaAegis.server_zoo import FedAvgServer, ParaAegisServer
from ParaAegis.protocol_zoo import fedavg, paraaegis
from ParaAegis.training import fetch_model, fetch_dataset, fetch_datasplitter
from ParaAegis.training import Trainer
from copy import deepcopy
import ray
import tenseal as ts
from ParaAegis.utils import find_dp_noise_multiplier
from ParaAegis.client_zoo.HeAvgClient import HeAvgClient
from ParaAegis.server_zoo import HeAvgServer

import logging
import swanlab



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
    LR = 0.001
    CLIP_THR = 10.0

    models = [fetch_model(MODEL_NAME).to(DEVICE) for _ in range(N_CLIENTS)]
    init_state = deepcopy(models[0].state_dict())
    
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
    
    # sk_bytes= he_ctx.serialize(save_secret_key=True, save_galois_keys=True)
    # pk_bytes=he_ctx.serialize(save_galois_keys=True)
    # <<=============End
    # clients = [HeAvgClient.remote(trainer=trainer, n_epochs=N_EPOCHS, ckks_bytes=sk_bytes) for trainer in trainers]
    # server = HeAvgServer.remote(ckks_bytes=pk_bytes)
    clients = [FedAvgClient.remote(trainer=trainer, n_epochs=N_EPOCHS) for trainer in trainers]
    server = FedAvgServer.remote()
    # noise_multiplier = find_dp_noise_multiplier(
    #     target_epsilon=5.0,  # 目标隐私预算
    #     delta=1e-5,          # 允许的失败概率
    #     n_steps=N_ROUNDS,    # 总轮数
    #     sensitivity=CLIP_THR / (len(train_subsets[0]))      # 梯度的敏感度
    # )
    
    # grad_len = len(trainers[0].state)
    
    # clients = [ParaAegisClient.remote(
    #     trainer=trainer,
    #     n_epochs=N_EPOCHS,
    #     he_ctx_bytes=sk_bytes,
    #     he_fraction=0.1,
    #     noise_multiplier=noise_multiplier,
    #     clip_thr=CLIP_THR
    # ) for trainer in trainers]
    
    # server = ParaAegisServer.remote(
    #     grad_len=grad_len,
    #     he_frac=0.1,
    #     he_ctx_bytes=pk_bytes
    # )

    # swanlab.init(
    #     # 设置将记录此次运行的项目信息
    #     project="ParaAegis",
    #     workspace="NaifenMizuha",
    #     # 跟踪超参数和运行元数据
    #     config={
    #         "model_name": MODEL_NAME,
    #         "dataset_name": DATASET_NAME,
    #         "split_name": SPLIT_NAME,
    #         "n_clients": N_CLIENTS,
    #         "n_epochs": N_EPOCHS,
    #         "n_rounds": N_ROUNDS,
    #         "lr": LR,
    #         "clip_thr": CLIP_THR,
    #         "destination": "para-aegis预实验"
    #     }
    # )

    result = fedavg(server, clients, N_ROUNDS)

    ray.shutdown()
    print(result)
    
    
