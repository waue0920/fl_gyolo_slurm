from sklearn.decomposition import PCA
import torch
from collections import OrderedDict, deque
import numpy as np
from sklearn.decomposition import PCA


def get_loss_drop(results):
    """
    計算 LossDrop，results 為 list of dict 或 numpy array，需包含 val_loss 欄位。
    例如 results = [{'val/box_loss': ...}, ...] 或 results = np.array([...])
    """
    if results is None or len(results) < 2:
        return 0.0
    # 假設 val/box_loss 欄位
    if isinstance(results, list):
        # 取最後兩個 epoch 的 val/box_loss
        losses = [r.get('val/box_loss', None) for r in results if 'val/box_loss' in r]
    elif isinstance(results, np.ndarray):
        # 假設第一欄是 val/box_loss
        losses = results[:, 0]
    else:
        return 0.0
    if len(losses) < 2 or losses[-2] is None or losses[-1] is None:
        return 0.0
    return float(losses[-2] - losses[-1])

def get_grad_var(weights_history):
    """
    計算 GradVariance，weights_history 為 list of OrderedDict，每個為一輪的 weights。
    """
    if weights_history is None or len(weights_history) < 2:
        return 0.0
    # 計算 weight delta
    deltas = []
    for i in range(1, len(weights_history)):
        delta = []
        for k in weights_history[i].keys():
            delta.append((weights_history[i][k] - weights_history[i-1][k]).cpu().numpy().flatten())
        delta = np.concatenate(delta)
        deltas.append(delta)
    deltas = np.stack(deltas)
    # 取所有 delta 的變異數
    return float(np.var(deltas))

def pca_reduce(X, n_components=8, solver='full'):
    """
    X: numpy array, shape [n_samples, n_features]
    n_components: int, target dimension
    solver: PCA svd_solver
    return: numpy array, shape [n_samples, n_components]
    """
    pca = PCA(n_components=n_components, svd_solver=solver)
    X_reduced = pca.fit_transform(X)
    return X_reduced

def softmax(x, temperature=1.0):
    x = x / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def aggregate(
    client_weights,
    client_vectors,
    global_weights,
):
    import os
    history_window = int(os.environ.get('SERVER_FEDAWA_HISTORY_WINDOW', 5))
    pca_dim = int(os.environ.get('SERVER_FEDAWA_PCA_DIM', 4))
    softmax_temperature = float(os.environ.get('SERVER_FEDAWA_SOFTMAX_TEMPERATURE', 1.0))
    lossdrop_weight = float(os.environ.get('SERVER_FEDAWA_LOSSDROP_WEIGHT', 1.0))
    gradvar_weight = float(os.environ.get('SERVER_FEDAWA_GRADVAR_WEIGHT', 1.0))
    pca_solver = os.environ.get('SERVER_FEDAWA_PCA_SOLVER', 'full')
    norm_eps = float(os.environ.get('SERVER_FEDAWA_NORM_EPS', 1e-8))
    num_clients = len(client_weights)
    # 1. 計算 Ucli_r = wi - global_weights
    Ucli_list = []
    for wi in client_weights:
        u = []
        for k in global_weights.keys():
            u.append((wi[k] - global_weights[k]).cpu().numpy().flatten())
        Ucli_list.append(np.concatenate(u))
    Ucli_arr = np.stack(Ucli_list)  # shape: [num_clients, total_params]
    # 2. 降維 (PCA) + Normalize
    n_components = min(pca_dim, Ucli_arr.shape[0], Ucli_arr.shape[1])
    Vcli_arr = pca_reduce(Ucli_arr, n_components=n_components, solver=pca_solver)
    Vcli_arr = Vcli_arr / (np.linalg.norm(Vcli_arr, axis=1, keepdims=True) + norm_eps)
    # 3. 歷史窗口平均 (假設 client_vectors 裡有歷史)
    Hcli_arr = []
    for i, vec in enumerate(client_vectors):
        history = vec.get('history', [])
        history = history[-history_window:] + [Vcli_arr[i]]
        h = np.mean(np.stack(history), axis=0)
        Hcli_arr.append(h)
        vec['history'] = history
    # 4. 拼接 LossDrop, GradVariance，並加權
    ClientVector_arr = []
    for i, vec in enumerate(client_vectors):
        loss_drop = get_loss_drop(vec.get('results', None)) if 'results' in vec else vec.get('loss_drop', 0.0)
        grad_var = get_grad_var(vec.get('weights_history', None)) if 'weights_history' in vec else vec.get('grad_var', 0.0)
        client_vec = np.concatenate([
            Hcli_arr[i],
            [lossdrop_weight * loss_drop],
            [gradvar_weight * grad_var]
        ])
        ClientVector_arr.append(client_vec)
    ClientVector_arr = np.stack(ClientVector_arr)  # shape: [num_clients, n_components+2]
    # 5. 計算聚合權重 ai (softmax)
    scores = np.linalg.norm(ClientVector_arr, axis=1)
    ai = softmax(scores, temperature=softmax_temperature)
    # 6. 聚合 Wglo_r = Σ(Wcli_r × ai)
    agg_weights = OrderedDict()
    for k in global_weights.keys():
        agg_weights[k] = torch.zeros_like(global_weights[k], dtype=torch.float32)
    for wi, a in zip(client_weights, ai):
        for k in agg_weights.keys():
            agg_weights[k] += a * wi[k]
    return agg_weights
