# sparse_gp_model.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ConstantKernel as C, WhiteKernel
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def build_sparse_gp_model(X_train, Y_train, cv=3, n_jobs=-1, verbose=1):
    """
    构建 + 训练稀疏高斯过程回归模型，并返回：
    - best_model: 含 scaler 的完整管道（做预测用）
    - scaler: 标准化器（对 X 做 transform）
    - gp_for_std: 单独的 GaussianProcessRegressor，用来给出 mean + std
    - grid_search: 网格搜索对象（里面有所有搜索信息）
    """

    # 1. 管道：标准化 + GP
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gaussian', GaussianProcessRegressor())
    ])

    # 2. 定义稀疏核函数集合（与你原脚本一致）
    kernels = [
        # 稀疏 RBF 核
        C(0.1, (1e-3, 1e1)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e1)),

        # 稀疏 Matern 核
        C(0.1, (1e-3, 1e1)) * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e1), nu=1.5),

        # 稀疏 RationalQuadratic 核
        C(0.1, (1e-3, 1e1)) * RationalQuadratic(length_scale=0.5, alpha=10, alpha_bounds=(1, 100)),

        # RBF + 白噪声
        C(0.1, (1e-3, 1e1)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e1)) +
        WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e1)),

        # Matern + 白噪声
        C(0.1, (1e-3, 1e1)) * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e1), nu=1.5) +
        WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e1)),

        # 非常稀疏 RBF
        C(0.01, (1e-4, 1e0)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10)),

        # 针对小数据的稀疏 Matern
        C(0.05, (1e-3, 1e0)) * Matern(length_scale=0.3, length_scale_bounds=(1e-2, 5), nu=0.5)
    ]

    # 3. 参数网格
    param_grid = {
        'gaussian__kernel': kernels,
        'gaussian__alpha': [1e-5, 1e-3],
        'gaussian__n_restarts_optimizer': [5, 10],
        'gaussian__normalize_y': [True]
    }

    # 4. 网格搜索
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose
    )

    print("\n[模型] 开始稀疏高斯过程网格搜索...")
    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_

    # 5. 抽取 GP 的最佳参数
    best_gaussian_params = {}
    for key, value in grid_search.best_params_.items():
        if key.startswith('gaussian__'):
            best_gaussian_params[key.replace('gaussian__', '')] = value

    print("[模型] 最佳高斯过程参数：", best_gaussian_params)

    # 6. 用最佳参数重新训练一个“只含 GP 的模型”以获取 mean + std
    scaler = best_model.named_steps['scaler']
    X_train_scaled = scaler.transform(X_train)

    gp_for_std = GaussianProcessRegressor(**best_gaussian_params)
    gp_for_std.fit(X_train_scaled, Y_train)

    return {
        "best_model": best_model,
        "scaler": scaler,
        "gp_for_std": gp_for_std,
        "grid_search": grid_search
    }
