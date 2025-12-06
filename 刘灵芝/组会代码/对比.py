import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import mode
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ----------------------
# 1. 基础C4.5决策树（修复索引匹配）
# ----------------------
class C45DecisionTree:
    def __init__(self, prune_threshold=3):
        self.root = None
        self.prune_threshold = prune_threshold

    def _entropy(self, y):
        y_clean = np.nan_to_num(y, nan=np.nanmean(y) if not np.isnan(y).all() else 0.0,
                                posinf=np.percentile(y[np.isfinite(y)], 99) if np.isfinite(y).any() else 0.0,
                                neginf=np.percentile(y[np.isfinite(y)], 1) if np.isfinite(y).any() else 0.0)
        classes, counts = np.unique(y_clean, return_counts=True)
        probs = counts / len(y_clean)
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log2(probs))

    def _information_gain(self, X, y, feature_idx):
        X_clean = np.nan_to_num(X, nan=np.nanmean(X, axis=0), posinf=np.percentile(X[np.isfinite(X)], 99),
                                neginf=np.percentile(X[np.isfinite(X)], 1))
        y_clean = np.nan_to_num(y, nan=np.nanmean(y) if not np.isnan(y).all() else 0.0,
                                posinf=np.percentile(y[np.isfinite(y)], 99) if np.isfinite(y).any() else 0.0,
                                neginf=np.percentile(y[np.isfinite(y)], 1) if np.isfinite(y).any() else 0.0)

        base_entropy = self._entropy(y_clean)
        feature_values = np.unique(X_clean[:, feature_idx])
        weighted_entropy = 0.0
        for val in feature_values:
            mask = X_clean[:, feature_idx] == val
            subset_y = y_clean[mask]
            if len(subset_y) == 0:
                continue
            weighted_entropy += (len(subset_y) / len(y_clean)) * self._entropy(subset_y)
        return base_entropy - weighted_entropy

    def _gain_ratio(self, X, y, feature_idx):
        gain = self._information_gain(X, y, feature_idx)
        X_clean = np.nan_to_num(X, nan=np.nanmean(X, axis=0), posinf=np.percentile(X[np.isfinite(X)], 99),
                                neginf=np.percentile(X[np.isfinite(X)], 1))
        feature_values = np.unique(X_clean[:, feature_idx])
        probs = np.array([len(X_clean[X_clean[:, feature_idx] == val]) / len(X_clean) for val in feature_values])

        probs = np.clip(probs, 1e-10, 1.0)
        split_info = -np.sum(probs * np.log2(probs))

        if split_info < 1e-10:
            return 1e-10
        return gain / split_info

    def _best_split(self, X, y, feature_indices):
        best_gain_ratio = -np.inf
        best_feature = None
        for idx in feature_indices:
            gain_ratio = self._gain_ratio(X, y, idx)
            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_feature = idx
        return best_feature

    def _build_tree(self, X, y, feature_indices, depth=0):
        # 修复核心：使用当前X/y的清洗结果，而非原始数据，确保维度匹配
        X_clean = np.nan_to_num(X, nan=np.nanmean(X, axis=0), posinf=np.percentile(X[np.isfinite(X)], 99),
                                neginf=np.percentile(X[np.isfinite(X)], 1))
        y_clean = np.nan_to_num(y, nan=np.nanmean(y) if not np.isnan(y).all() else 0.0,
                                posinf=np.percentile(y[np.isfinite(y)], 99) if np.isfinite(y).any() else 0.0,
                                neginf=np.percentile(y[np.isfinite(y)], 1) if np.isfinite(y).any() else 0.0)

        if len(y_clean) <= self.prune_threshold:
            return {"type": "leaf", "value": np.mean(y_clean)}
        if np.max(y_clean) - np.min(y_clean) < 1e-6:
            return {"type": "leaf", "value": y_clean[0]}
        if len(feature_indices) == 0:
            return {"type": "leaf", "value": np.mean(y_clean)}

        best_feature = self._best_split(X_clean, y_clean, feature_indices)
        if best_feature is None:
            return {"type": "leaf", "value": np.mean(y_clean)}

        tree = {"type": "node", "feature": best_feature, "children": {}}
        feature_values = np.unique(X_clean[:, best_feature])
        remaining_features = [f for f in feature_indices if f != best_feature]
        for val in feature_values:
            mask = X_clean[:, best_feature] == val  # 基于当前X_clean的mask，维度匹配
            if len(X[mask]) == 0:  # X是当前层级的数据集，与mask维度一致
                tree["children"][val] = {"type": "leaf", "value": np.mean(y_clean)}
                continue
            # 递归传入当前层级的子集，而非原始数据
            tree["children"][val] = self._build_tree(X[mask], y[mask], remaining_features, depth + 1)
        return tree

    def fit(self, X, y):
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X必须是2维数组，y必须是1维数组")
        feature_indices = list(range(X.shape[1]))
        self.root = self._build_tree(X, y, feature_indices)

    def _predict_sample(self, x, node):
        if node["type"] == "leaf":
            return node["value"]
        x_clean = np.nan_to_num(x, nan=0.0, posinf=1e20, neginf=-1e20)
        feature_val = x_clean[node["feature"]]
        if feature_val not in node["children"]:
            def collect_leaf_values(current_node):
                leaf_values = []
                if current_node["type"] == "leaf":
                    leaf_values.append(current_node["value"])
                else:
                    for child in current_node["children"].values():
                        leaf_values.extend(collect_leaf_values(child))
                return leaf_values

            all_leaf_values = collect_leaf_values(node)
            return np.mean(all_leaf_values) if all_leaf_values else 0.0
        return self._predict_sample(x, node["children"][feature_val])

    def predict(self, X):
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e20, neginf=-1e20)
        return np.array([self._predict_sample(x, self.root) for x in X_clean])


# ----------------------
# 2. 三个对比算法实现（修复抽样逻辑+缩进）
# ----------------------
# 2.1 普通随机森林（等权平均）
class SimpleRandomForestRegressor:
    def __init__(self, n_estimators=50, prune_threshold=3, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.prune_threshold = prune_threshold
        self.max_features = max_features
        self.trees_shear = []
        self.trees_peel = []

    def _bootstrap(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y_shear, y_peel):
        n_features = X.shape[1]
        if self.max_features == "sqrt":
            self.m = max(2, min(int(np.sqrt(n_features)), n_features))
        else:
            self.m = max(2, min(self.max_features, n_features))

        for _ in range(self.n_estimators):
            X_train_shear, y_train_shear = self._bootstrap(X, y_shear)
            X_train_peel, y_train_peel = self._bootstrap(X, y_peel)
            feature_indices = np.random.choice(n_features, size=self.m, replace=False)

            tree_shear = C45DecisionTree(prune_threshold=self.prune_threshold)
            tree_shear.fit(X_train_shear[:, feature_indices], y_train_shear)
            self.trees_shear.append((tree_shear, feature_indices))

            tree_peel = C45DecisionTree(prune_threshold=self.prune_threshold)
            tree_peel.fit(X_train_peel[:, feature_indices], y_train_peel)
            self.trees_peel.append((tree_peel, feature_indices))

    def predict(self, X):
        shear_preds = [tree.predict(X[:, idx]) for tree, idx in self.trees_shear]
        peel_preds = [tree.predict(X[:, idx]) for tree, idx in self.trees_peel]

        pred_shear = np.mean(np.array(shear_preds), axis=0).flatten()
        pred_peel = np.mean(np.array(peel_preds), axis=0).flatten()

        pred_shear = np.nan_to_num(pred_shear, nan=0.0, posinf=1e20, neginf=-1e20)
        pred_peel = np.nan_to_num(pred_peel, nan=0.0, posinf=1e20, neginf=-1e20)
        return pred_shear, pred_peel


# 2.2 加权随机森林（修复抽样维度+缩进问题）
class WeightedRandomForestRegressor:
    def __init__(self, n_estimators=50, prune_threshold=3, max_features="sqrt", val_split_ratio=0.1):
        self.n_estimators = n_estimators
        self.prune_threshold = prune_threshold
        self.max_features = max_features
        self.val_split_ratio = val_split_ratio
        self.trees_shear = []
        self.trees_peel = []
        self.weights_shear = []
        self.weights_peel = []

    def _bootstrap_with_val(self, X, y_shear, y_peel):
        """修复：确保训练集和验证集索引基于同一原始数据，维度一致"""
        n_samples = X.shape[0]
        if n_samples < 8:
            raise ValueError(f"样本数过少（仅 {n_samples} 个）")

        # 修复1：训练集和验证集索引不重叠，且基于原始数据维度
        val_size = max(int(n_samples * self.val_split_ratio), 3)
        val_indices = np.random.choice(n_samples, size=val_size, replace=False)
        train_indices = np.array([i for i in range(n_samples) if i not in val_indices])

        # 修复2：小样本时训练集补充抽样到与原始样本数一致（避免维度过小）
        if len(train_indices) < n_samples:
            补充_indices = np.random.choice(n_samples, size=n_samples - len(train_indices), replace=False)
            train_indices = np.concatenate([train_indices, 补充_indices])

        return (X[train_indices], y_shear[train_indices], y_peel[train_indices],
                X[val_indices], y_shear[val_indices], y_peel[val_indices])

    def _calculate_weight(self, y_true, y_pred):
        y_true_clean = np.nan_to_num(y_true, nan=np.nanmean(y_true) if not np.isnan(y_true).all() else 0.0,
                                     posinf=np.percentile(y_true[np.isfinite(y_true)], 99) if np.isfinite(
                                         y_true).any() else 0.0,
                                     neginf=np.percentile(y_true[np.isfinite(y_true)], 1) if np.isfinite(
                                         y_true).any() else 0.0)
        y_pred_clean = np.nan_to_num(y_pred, nan=np.nanmean(y_pred) if not np.isnan(y_pred).all() else 0.0,
                                     posinf=np.percentile(y_pred[np.isfinite(y_pred)], 99) if np.isfinite(
                                         y_pred).any() else 0.0,
                                     neginf=np.percentile(y_pred[np.isfinite(y_pred)], 1) if np.isfinite(
                                         y_pred).any() else 0.0)

        if np.allclose(y_pred_clean, y_pred_clean[0]) or np.allclose(y_true_clean, y_true_clean[0]):
            return 0.1

        try:
            r2 = r2_score(y_true_clean, y_pred_clean)
            r2 = np.clip(r2, -1.0, 1.0)
            return max(r2, 0.1)
        except Exception as e:
            return 0.1

    def fit(self, X, y_shear, y_peel):
        n_features = X.shape[1]
        if self.max_features == "sqrt":
            self.m = max(2, min(int(np.sqrt(n_features)), n_features))
        else:
            self.m = max(2, min(self.max_features, n_features))

        for _ in range(self.n_estimators):
            try:
                X_train, y_shear_train, y_peel_train, X_val, y_shear_val, y_peel_val = self._bootstrap_with_val(X,
                                                                                                                y_shear,
                                                                                                                y_peel)
                feature_indices = np.random.choice(n_features, size=self.m, replace=False)

                tree_shear = C45DecisionTree(prune_threshold=self.prune_threshold)
                tree_shear.fit(X_train[:, feature_indices], y_shear_train)
                y_shear_val_pred = tree_shear.predict(X_val[:, feature_indices])
                weight_shear = self._calculate_weight(y_shear_val, y_shear_val_pred)
                self.trees_shear.append((tree_shear, feature_indices))
                self.weights_shear.append(weight_shear)

                tree_peel = C45DecisionTree(prune_threshold=self.prune_threshold)
                tree_peel.fit(X_train[:, feature_indices], y_peel_train)
                y_peel_val_pred = tree_peel.predict(X_val[:, feature_indices])
                weight_peel = self._calculate_weight(y_peel_val, y_peel_val_pred)
                self.trees_peel.append((tree_peel, feature_indices))
                self.weights_peel.append(weight_peel)
            except Exception as e:
                print(f"警告：单棵树训练失败，跳过该树。错误：{str(e)[:100]}")
                continue

        sum_shear = np.sum(self.weights_shear) + 1e-10
        sum_peel = np.sum(self.weights_peel) + 1e-10
        self.weights_shear = np.array(self.weights_shear) / sum_shear
        self.weights_peel = np.array(self.weights_peel) / sum_peel

    def predict(self, X):
        shear_preds = [tree.predict(X[:, idx]) * w for (tree, idx), w in zip(self.trees_shear, self.weights_shear)]
        peel_preds = [tree.predict(X[:, idx]) * w for (tree, idx), w in zip(self.trees_peel, self.weights_peel)]

        pred_shear = np.sum(np.array(shear_preds), axis=0).flatten()
        pred_peel = np.sum(np.array(peel_preds), axis=0).flatten()

        pred_shear = np.nan_to_num(pred_shear, nan=0.0, posinf=1e20, neginf=-1e20)
        pred_peel = np.nan_to_num(pred_peel, nan=0.0, posinf=1e20, neginf=-1e20)
        return pred_shear, pred_peel


# 2.3 PSO优化加权随机森林（复用修复后的加权随机森林）
class PSOOptimizer:
    def __init__(self, X_train, y_shear_train, y_peel_train, X_val, y_shear_val, y_peel_val, param_bounds,
                 n_particles=6, max_iter=10):
        self.X_train = X_train
        self.y_shear_train = y_shear_train
        self.y_peel_train = y_peel_train
        self.X_val = X_val
        self.y_shear_val = y_shear_val
        self.y_peel_val = y_peel_val
        self.param_bounds = param_bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5

    def _fitness(self, params):
        try:
            n_estimators = int(round(params[0]))
            prune_threshold = int(round(params[1]))
            val_split_ratio = params[2]
            max_features = int(round(params[3]))

            n_estimators = max(10, min(50, n_estimators))
            prune_threshold = max(2, min(5, prune_threshold))
            val_split_ratio = max(0.05, min(0.15, val_split_ratio))
            max_features = max(2, min(self.X_train.shape[1], max_features))

            wrf = WeightedRandomForestRegressor(
                n_estimators=n_estimators,
                prune_threshold=prune_threshold,
                val_split_ratio=val_split_ratio,
                max_features=max_features
            )
            wrf.fit(self.X_train, self.y_shear_train, self.y_peel_train)

            pred_shear_val, pred_peel_val = wrf.predict(self.X_val)

            y_shear_val_clean = np.nan_to_num(self.y_shear_val, nan=np.nanmean(self.y_shear_val) if not np.isnan(
                self.y_shear_val).all() else 0.0,
                                              posinf=np.percentile(self.y_shear_val[np.isfinite(self.y_shear_val)],
                                                                   99) if np.isfinite(self.y_shear_val).any() else 0.0,
                                              neginf=np.percentile(self.y_shear_val[np.isfinite(self.y_shear_val)],
                                                                   1) if np.isfinite(self.y_shear_val).any() else 0.0)
            y_peel_val_clean = np.nan_to_num(self.y_peel_val, nan=np.nanmean(self.y_peel_val) if not np.isnan(
                self.y_peel_val).all() else 0.0,
                                             posinf=np.percentile(self.y_peel_val[np.isfinite(self.y_peel_val)],
                                                                  99) if np.isfinite(self.y_peel_val).any() else 0.0,
                                             neginf=np.percentile(self.y_peel_val[np.isfinite(self.y_peel_val)],
                                                                  1) if np.isfinite(self.y_peel_val).any() else 0.0)

            r2_shear = r2_score(y_shear_val_clean, pred_shear_val)
            r2_peel = r2_score(y_peel_val_clean, pred_peel_val)
            fitness = r2_shear + r2_peel

            return fitness if not (np.isnan(fitness) or np.isinf(fitness)) else -1e9
        except Exception as e:
            print(f"警告：参数组合 {params.round(2)} 训练失败，适应度设为-1e9。错误：{str(e)[:150]}")
            return -1e9

    def _initialize_particles(self):
        n_params = len(self.param_bounds)
        particles = np.zeros((self.n_particles, n_params))
        for i in range(n_params):
            particles[:, i] = np.random.uniform(self.param_bounds[i][0], self.param_bounds[i][1], self.n_particles)
        velocities = np.random.uniform(-0.5, 0.5, (self.n_particles, n_params))
        return particles, velocities

    def optimize(self):
        particles, velocities = self._initialize_particles()
        p_best = particles.copy()
        p_best_fitness = np.array([self._fitness(p) for p in particles])
        g_best_idx = np.argmax(p_best_fitness)
        g_best = particles[g_best_idx].copy()
        g_best_fitness = p_best_fitness[g_best_idx]

        for iter in range(self.max_iter):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                        self.w * velocities[i]
                        + self.c1 * r1 * (p_best[i] - particles[i])
                        + self.c2 * r2 * (g_best - particles[i])
                )
                particles[i] += velocities[i]
                for j in range(len(self.param_bounds)):
                    particles[i][j] = max(self.param_bounds[j][0], min(self.param_bounds[j][1], particles[i][j]))
                current_fitness = self._fitness(particles[i])
                if current_fitness > p_best_fitness[i]:
                    p_best[i] = particles[i].copy()
                    p_best_fitness[i] = current_fitness
                if current_fitness > g_best_fitness:
                    g_best = particles[i].copy()
                    g_best_fitness = current_fitness

            print(f"PSO迭代{iter + 1}/{self.max_iter}，当前最优适应度：{g_best_fitness:.4f}")

        return g_best, g_best_fitness


class PSOWRFRegressor:
    def __init__(self, param_bounds=None, pso_kwargs=None):
        self.param_bounds = param_bounds or [
            (10, 50),  # 决策树棵数
            (2, 5),  # 剪枝阈值
            (0.05, 0.15),  # 预测试样本占比
            (2, 5)  # 随机特征数
        ]
        self.pso_kwargs = pso_kwargs or {"n_particles": 6, "max_iter": 10}
        self.best_params = None
        self.wrf = None

    def fit(self, X, y_shear, y_peel):
        total_samples = len(X)
        test_size = 0.1 if total_samples <= 20 else 0.15 if total_samples <= 50 else 0.2

        X_train, X_val, y_shear_train, y_shear_val, y_peel_train, y_peel_val = train_test_split(
            X, y_shear, y_peel, test_size=test_size, random_state=42, shuffle=True
        )

        if len(X_train) < 10 or len(X_val) < 2:
            raise ValueError(f"拆分后样本数不足（训练集 {len(X_train)} 个，验证集 {len(X_val)} 个）")

        print(f"PSO优化 - 训练集：{len(X_train)} 个样本，验证集：{len(X_val)} 个样本")

        pso = PSOOptimizer(
            X_train, y_shear_train, y_peel_train, X_val, y_shear_val, y_peel_val,
            self.param_bounds, **self.pso_kwargs
        )
        self.best_params, best_fitness = pso.optimize()

        n_estimators = int(round(self.best_params[0]))
        prune_threshold = int(round(self.best_params[1]))
        val_split_ratio = self.best_params[2]
        max_features = int(round(self.best_params[3]))

        self.wrf = WeightedRandomForestRegressor(
            n_estimators=n_estimators,
            prune_threshold=prune_threshold,
            val_split_ratio=val_split_ratio,
            max_features=max_features
        )
        self.wrf.fit(X, y_shear, y_peel)

        print("\n=== PSO优化后的最优参数 ===")
        print(f"决策树棵数：{n_estimators}")
        print(f"剪枝阈值：{prune_threshold}")
        print(f"预测试样本占比：{val_split_ratio:.2f}")
        print(f"随机特征数：{max_features}")

    def predict(self, X):
        return self.wrf.predict(X)


# ----------------------
# 3. 模型评估与可视化工具（保持不变）
# ----------------------
def evaluate_model(y_true, y_pred, model_name, target_name):
    y_true_clean = np.nan_to_num(y_true, nan=np.nanmean(y_true) if not np.isnan(y_true).all() else 0.0,
                                 posinf=np.percentile(y_true[np.isfinite(y_true)], 99) if np.isfinite(
                                     y_true).any() else 0.0,
                                 neginf=np.percentile(y_true[np.isfinite(y_true)], 1) if np.isfinite(
                                     y_true).any() else 0.0)

    r2 = r2_score(y_true_clean, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred))
    mae = mean_absolute_error(y_true_clean, y_pred)

    return {
        "模型": model_name,
        "目标变量": target_name,
        "R²": round(r2, 3),
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3)
    }


def plot_performance_comparison(results):
    models = list(set([r["模型"] for r in results]))
    targets = ["剪切强度", "扯离强度"]
    metrics = ["R²", "RMSE", "MAE"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("三个算法性能指标对比", fontsize=16, fontweight='bold')

    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(targets))
        width = 0.25

        for i, model in enumerate(models):
            values = [next(r[metric] for r in results if r["模型"] == model and r["目标变量"] == t) for t in targets]
            ax.bar(x + i * width, values, width, label=model, color=colors[i], alpha=0.8)

        ax.set_xlabel("目标变量", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} 对比", fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(targets)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("算法性能对比图.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_pred_true_scatter(y_true_shear, y_true_peel, preds_dict):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("预测值 vs 真实值散点图", fontsize=16, fontweight='bold')

    models = list(preds_dict.keys())
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for row, (y_true, target_name) in enumerate(zip([y_true_shear, y_true_peel], ["剪切强度", "扯离强度"])):
        y_true_clean = np.nan_to_num(y_true, nan=np.nanmean(y_true) if not np.isnan(y_true).all() else 0.0,
                                     posinf=np.percentile(y_true[np.isfinite(y_true)], 99) if np.isfinite(
                                         y_true).any() else 0.0,
                                     neginf=np.percentile(y_true[np.isfinite(y_true)], 1) if np.isfinite(
                                         y_true).any() else 0.0)

        for col, (model, color) in enumerate(zip(models, colors)):
            ax = axes[row, col]
            y_pred = preds_dict[model][row]

            ax.scatter(y_true_clean, y_pred, color=color, alpha=0.6, s=50)

            min_val = min(y_true_clean.min(), y_pred.min())
            max_val = max(y_true_clean.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="理想预测线")

            r2 = r2_score(y_true_clean, y_pred)
            ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=11)

            ax.set_xlabel(f"真实{target_name}", fontsize=12)
            ax.set_ylabel(f"预测{target_name}", fontsize=12)
            ax.set_title(f"{model} - {target_name}", fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("预测值vs真实值散点图.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_residual_distribution(y_true_shear, y_true_peel, preds_dict):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("残差分布图（残差=真实值-预测值）", fontsize=16, fontweight='bold')

    models = list(preds_dict.keys())
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for row, (y_true, target_name) in enumerate(zip([y_true_shear, y_true_peel], ["剪切强度", "扯离强度"])):
        y_true_clean = np.nan_to_num(y_true, nan=np.nanmean(y_true) if not np.isnan(y_true).all() else 0.0,
                                     posinf=np.percentile(y_true[np.isfinite(y_true)], 99) if np.isfinite(
                                         y_true).any() else 0.0,
                                     neginf=np.percentile(y_true[np.isfinite(y_true)], 1) if np.isfinite(
                                         y_true).any() else 0.0)

        for col, (model, color) in enumerate(zip(models, colors)):
            ax = axes[row, col]
            y_pred = preds_dict[model][row]
            residuals = y_true_clean - y_pred

            ax.hist(residuals, bins=15, color=color, alpha=0.6, density=True, label="残差分布")
            ax.axvline(x=0, color='k', linestyle='--', lw=2, label="残差=0")

            res_mean = residuals.mean()
            ax.text(0.05, 0.95, f"残差均值 = {res_mean:.3f}", transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=11)

            ax.set_xlabel(f"{target_name}残差", fontsize=12)
            ax.set_ylabel("密度", fontsize=12)
            ax.set_title(f"{model} - {target_name}", fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("残差分布图.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(models, X, feature_names, y_shear, y_peel):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    importances_list = []
    model_names = []

    for model_name, model in models.items():
        if hasattr(model, 'wrf'):
            forest = model.wrf
        else:
            forest = model

        n_features = X.shape[1]
        importances = np.zeros(n_features)
        X_clean = np.nan_to_num(X, nan=np.nanmean(X, axis=0), posinf=np.percentile(X[np.isfinite(X)], 99),
                                neginf=np.percentile(X[np.isfinite(X)], 1))

        pred_shear, pred_peel = forest.predict(X_clean)
        y_shear_clean = np.nan_to_num(y_shear, nan=np.nanmean(y_shear) if not np.isnan(y_shear).all() else 0.0,
                                      posinf=np.percentile(y_shear[np.isfinite(y_shear)], 99) if np.isfinite(
                                          y_shear).any() else 0.0,
                                      neginf=np.percentile(y_shear[np.isfinite(y_shear)], 1) if np.isfinite(
                                          y_shear).any() else 0.0)
        y_peel_clean = np.nan_to_num(y_peel, nan=np.nanmean(y_peel) if not np.isnan(y_peel).all() else 0.0,
                                     posinf=np.percentile(y_peel[np.isfinite(y_peel)], 99) if np.isfinite(
                                         y_peel).any() else 0.0,
                                     neginf=np.percentile(y_peel[np.isfinite(y_peel)], 1) if np.isfinite(
                                         y_peel).any() else 0.0)
        r2_original = (r2_score(y_shear_clean, pred_shear) + r2_score(y_peel_clean, pred_peel)) / 2

        for i in range(n_features):
            X_shuffled = X_clean.copy()
            np.random.shuffle(X_shuffled[:, i])
            pred_shear_shuf, pred_peel_shuf = forest.predict(X_shuffled)
            r2_shuf = (r2_score(y_shear_clean, pred_shear_shuf) + r2_score(y_peel_clean, pred_peel_shuf)) / 2
            importances[i] = r2_original - r2_shuf

        importances = importances / (importances.sum() + 1e-10)
        importances_list.append(importances)
        model_names.append(model_name)

    x = np.arange(len(feature_names))
    width = 0.25
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, (importances, model_name, color) in enumerate(zip(importances_list, model_names, colors)):
        ax.bar(x + i * width, importances, width, label=model_name, color=color, alpha=0.8)

    ax.set_xlabel("特征", fontsize=12)
    ax.set_ylabel("归一化重要性", fontsize=12)
    ax.set_title("三个算法的特征重要性对比", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("特征重要性对比图.png", dpi=300, bbox_inches='tight')
    plt.show()


# ----------------------
# 4. 主函数（修复特征重要性传参）
# ----------------------
def main():
    # 1. 读取原始数据（不清洗）
    excel_path = "模拟数据.xlsx"
    try:
        data = pd.read_excel(excel_path, sheet_name=0)
        print("Excel数据读取成功！数据形状：", data.shape)
    except Exception as e:
        print(f"Excel读取失败：{e}")
        return

    # 2. 提取数据（不清洗）
    feature_cols = ['环境温度(℃)', '环境湿度(%)', '固化温度(℃)', '固化时间(h)', '推进剂浇筑温度(℃)']
    target_shear_col = '剪切强度(MPa)'
    target_peel_col = '扯离强度(kN/m)'

    X = data[feature_cols].values
    y_shear = data[target_shear_col].values
    y_peel = data[target_peel_col].values

    print(f"\n提取后数据形状：X={X.shape}, y_shear={y_shear.shape}, y_peel={y_peel.shape}")
    print(f"原始数据校验：")
    print(f"X 中 NaN 数量：{np.isnan(X).sum()}, inf 数量：{np.isinf(X).sum()}")
    print(f"y_shear 中 NaN 数量：{np.isnan(y_shear).sum()}, inf 数量：{np.isinf(y_shear).sum()}")
    print(f"y_peel 中 NaN 数量：{np.isnan(y_peel).sum()}, inf 数量：{np.isinf(y_peel).sum()}")

    # 3. 初始化三个对比算法
    print("\n=== 初始化三个对比算法 ===")
    simple_rf = SimpleRandomForestRegressor(
        n_estimators=30, prune_threshold=3, max_features=3
    )
    weighted_rf = WeightedRandomForestRegressor(
        n_estimators=30, prune_threshold=3, max_features=3, val_split_ratio=0.1
    )
    psowrf = PSOWRFRegressor(
        param_bounds=[(10, 50), (2, 5), (0.05, 0.15), (2, 5)],
        pso_kwargs={"n_particles": 6, "max_iter": 10}
    )

    # 4. 训练三个算法
    print("\n=== 训练普通随机森林 ===")
    simple_rf.fit(X, y_shear, y_peel)

    print("\n=== 训练加权随机森林 ===")
    weighted_rf.fit(X, y_shear, y_peel)

    print("\n=== 训练PSO优化加权随机森林 ===")
    try:
        psowrf.fit(X, y_shear, y_peel)
    except Exception as e:
        print(f"PSOWRF训练失败：{e}")
        return

    # 5. 三个算法预测
    print("\n=== 三个算法预测 ===")
    sr_shear, sr_peel = simple_rf.predict(X)
    wr_shear, wr_peel = weighted_rf.predict(X)
    pw_shear, pw_peel = psowrf.predict(X)

    preds_dict = {
        "普通随机森林": [sr_shear, sr_peel],
        "加权随机森林": [wr_shear, wr_peel],
        "PSO优化加权随机森林": [pw_shear, pw_peel]
    }

    # 6. 模型评估
    print("\n=== 模型评估结果 ===")
    results = []
    results.append(evaluate_model(y_shear, sr_shear, "普通随机森林", "剪切强度"))
    results.append(evaluate_model(y_peel, sr_peel, "普通随机森林", "扯离强度"))
    results.append(evaluate_model(y_shear, wr_shear, "加权随机森林", "剪切强度"))
    results.append(evaluate_model(y_peel, wr_peel, "加权随机森林", "扯离强度"))
    results.append(evaluate_model(y_shear, pw_shear, "PSO优化加权随机森林", "剪切强度"))
    results.append(evaluate_model(y_peel, pw_peel, "PSO优化加权随机森林", "扯离强度"))

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # 7. 可视化对比（修复特征重要性传参）
    print("\n=== 生成可视化图表 ===")
    plot_performance_comparison(results)
    plot_pred_true_scatter(y_shear, y_peel, preds_dict)
    plot_residual_distribution(y_shear, y_peel, preds_dict)

    models_for_importance = {
        "普通随机森林": simple_rf,
        "加权随机森林": weighted_rf,
        "PSO优化加权随机森林": psowrf
    }
    plot_feature_importance(models_for_importance, X, feature_cols, y_shear, y_peel)

    # 8. 最优温度搜索
    print("\n=== PSO优化加权随机森林 - 最优温度条件搜索 ===")
    env_temps = np.linspace(20, 35, 16)
    cure_temps = np.linspace(55, 70, 16)
    pour_temps = np.linspace(50, 65, 16)
    fixed_humidity = 50
    fixed_cure_time = 4.5

    best_combined = 0
    best_temp_params = None
    best_shear = 0.0
    best_peel = 0.0

    for env in env_temps:
        for cure in cure_temps:
            for pour in pour_temps:
                input_data = np.array([[env, fixed_humidity, cure, fixed_cure_time, pour]])
                pred_shear, pred_peel = psowrf.predict(input_data)
                shear_val = pred_shear[0] if not (np.isnan(pred_shear[0]) or np.isinf(pred_shear[0])) else 0.0
                peel_val = pred_peel[0] if not (np.isnan(pred_peel[0]) or np.isinf(pred_peel[0])) else 0.0
                combined_strength = shear_val + peel_val

                if combined_strength > best_combined:
                    best_combined = combined_strength
                    best_shear = shear_val
                    best_peel = peel_val
                    best_temp_params = (env, cure, pour)

    print("\n=== 最优温度条件 ===")
    print(f"环境温度：{best_temp_params[0]:.1f}℃")
    print(f"固化温度：{best_temp_params[1]:.1f}℃")
    print(f"推进剂浇筑温度：{best_temp_params[2]:.1f}℃")
    print(f"最优剪切强度：{best_shear:.3f}MPa")
    print(f"最优扯离强度：{best_peel:.3f}kN/m")
    print(f"综合强度（剪切+扯离）：{best_combined:.3f}")


if __name__ == "__main__":
    main()