import torch
from collections import OrderedDict
import numpy as np

class FedOptState:
    def __init__(self, global_weights, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = OrderedDict({k: torch.zeros_like(v) for k, v in global_weights.items()})
        self.v = OrderedDict({k: torch.zeros_like(v) for k, v in global_weights.items()})
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

# 支持 FedAdam

def aggregate(client_weights, global_weights, optimizer_state=None):
    import os
    lr = float(os.environ.get('SERVER_FEDOPT_LR', 0.001))
    beta1 = float(os.environ.get('SERVER_FEDOPT_BETA1', 0.9))
    beta2 = float(os.environ.get('SERVER_FEDOPT_BETA2', 0.999))
    eps = float(os.environ.get('SERVER_FEDOPT_EPS', 1e-8))
    num_clients = len(client_weights)
    # 1. 計算每個 client 的 delta
    delta_list = []
    for wi in client_weights:
        delta = OrderedDict()
        for k in global_weights.keys():
            delta[k] = wi[k] - global_weights[k]
        delta_list.append(delta)
    # 2. 平均 delta
    avg_delta = OrderedDict()
    for k in global_weights.keys():
        avg_delta[k] = sum([delta[k] for delta in delta_list]) / num_clients
    # 3. 更新 FedAdam 狀態
    if optimizer_state is None:
        optimizer_state = FedOptState(global_weights, lr, beta1, beta2, eps)
    m = optimizer_state.m
    v = optimizer_state.v
    # 4. FedAdam 公式
    for k in global_weights.keys():
        m[k] = beta1 * m[k] + (1 - beta1) * avg_delta[k]
        v[k] = beta2 * v[k] + (1 - beta2) * (avg_delta[k] ** 2)
        global_weights[k] = global_weights[k] - lr * m[k] / (torch.sqrt(v[k]) + eps)
    # 5. 回傳新 weights 及 optimizer_state
    return global_weights, optimizer_state
