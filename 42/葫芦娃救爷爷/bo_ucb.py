# bo_ucb.py
import numpy as np


def suggest_next_x_ucb(
    gaussian_model,
    scaler,
    X_train_original,
    n_candidates=1000,
    kappa=2.0,
    mode="ucb",
    max_sigma=None,
    random_state=None
):
    """
    使用 高斯过程 + UCB/稳健目标，推荐下一组实验点 X。

    参数：
    - gaussian_model: 训练好的 GaussianProcessRegressor（gp_for_std）
    - scaler: 标准化器（与训练时一致）
    - X_train_original: 原始空间的 X_train（未标准化）
    - n_candidates: 在可行空间中随机采样的候选点数量
    - kappa: UCB 或稳健目标中的系数
    - mode:
        "ucb"    -> mu + kappa * sigma  （偏探索）
        "robust" -> mu - kappa * sigma  （偏重稳定、方差小）
    - max_sigma: 若不为 None，则只保留 sigma <= max_sigma 的候选点

    返回：
    - best_x: 推荐的下一组 X（原始尺度）
    - best_mu: 在该点的预测均值
    - best_sigma: 在该点的预测标准差
    """

    if random_state is not None:
        np.random.seed(random_state)

    n_features = X_train_original.shape[1]

    # 1. 用当前训练数据的 min/max 作为搜索空间
    x_min = X_train_original.min(axis=0)
    x_max = X_train_original.max(axis=0)

    # 2. 随机生成候选点
    candidates = np.random.uniform(
        low=x_min,
        high=x_max,
        size=(n_candidates, n_features)
    )

    # 3. 标准化后丢进 GP 预测
    candidates_scaled = scaler.transform(candidates)
    mu, sigma = gaussian_model.predict(candidates_scaled, return_std=True)

    # 4. 如有需要，过滤掉不确定度太大的点（为了稳定）
    if max_sigma is not None:
        mask = sigma <= max_sigma
        if mask.sum() > 0:
            candidates = candidates[mask]
            mu = mu[mask]
            sigma = sigma[mask]

    # 5. 计算目标函数
    if mode == "ucb":
        score = mu + kappa * sigma
    elif mode == "robust":
        score = mu - kappa * sigma
    else:
        raise ValueError("mode must be 'ucb' or 'robust'")

    # 6. 选择得分最高的点
    best_idx = np.argmax(score)
    best_x = candidates[best_idx]
    best_mu = mu[best_idx]
    best_sigma = sigma[best_idx]

    return best_x, best_mu, best_sigma
