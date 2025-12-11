import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import mode


# ----------------------
# 1. C4.5决策树类（小样本适配）
# ----------------------
class C45DecisionTree:
    def __init__(self, prune_threshold=3):  # 剪枝阈值从5→3，适配小样本
        self.root = None
        self.prune_threshold = prune_threshold

    def _entropy(self, y):
        # 临时清洗输入（不修改原始数据，仅用于计算）
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
        X_clean = np.nan_to_num(X, nan=np.nanmean(X, axis=0), posinf=np.percentile(X[np.isfinite(X)], 99),
                                neginf=np.percentile(X[np.isfinite(X)], 1))
        y_clean = np.nan_to_num(y, nan=np.nanmean(y) if not np.isnan(y).all() else 0.0,
                                posinf=np.percentile(y[np.isfinite(y)], 99) if np.isfinite(y).any() else 0.0,
                                neginf=np.percentile(y[np.isfinite(y)], 1) if np.isfinite(y).any() else 0.0)

        # 调整终止条件，适配小样本（样本数≥2即可）
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
            mask = X_clean[:, best_feature] == val
            if len(X_clean[mask]) == 0:
                tree["children"][val] = {"type": "leaf", "value": np.mean(y_clean)}
                continue
            tree["children"][val] = self._build_tree(
                X[mask], y[mask], remaining_features, depth + 1
            )
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
# 2. 加权随机森林回归类（小样本适配）
# ----------------------
class WeightedRandomForestRegressor:
    def __init__(self, n_estimators=50, prune_threshold=3, max_features="sqrt", val_split_ratio=0.1):
        self.n_estimators = n_estimators  # 树数从100→50，减少过拟合
        self.prune_threshold = prune_threshold
        self.max_features = max_features
        self.val_split_ratio = val_split_ratio  # 预测试占比从0.2→0.1，适配小样本
        self.trees_shear = []
        self.trees_peel = []
        self.weights_shear = []
        self.weights_peel = []

    def _bootstrap_with_val(self, X, y_shear, y_peel):
        n_samples = X.shape[0]
        # 小样本适配：最小样本数从10→8
        if n_samples < 8:
            raise ValueError(f"样本数过少（仅 {n_samples} 个），无法拆分训练集和验证集")

        # Bootstrap抽样：小样本时训练集不重复抽样（避免数据失真）
        if n_samples <= 30:
            train_indices = np.random.choice(n_samples, size=n_samples, replace=False)
        else:
            train_indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # 验证集最小 size 从5→3（适配小样本）
        val_size = max(int(n_samples * self.val_split_ratio), 3)
        val_indices = np.random.choice(n_samples, size=val_size, replace=False)

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

        if np.allclose(y_pred_clean, y_pred_clean[0]):
            return 0.1
        if np.allclose(y_true_clean, y_true_clean[0]):
            return 0.1

        try:
            r2 = r2_score(y_true_clean, y_pred_clean)
            r2 = np.clip(r2, -1.0, 1.0)
            return max(r2, 0.1)
        except Exception as e:
            return 0.1

    def fit(self, X, y_shear, y_peel):
        self.trees_shear = []
        self.trees_peel = []
        self.weights_shear = []
        self.weights_peel = []
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

        if len(self.weights_shear) == 0:
            raise ValueError("所有剪切强度树训练失败，请检查数据或参数")
        if len(self.weights_peel) == 0:
            raise ValueError("所有扯离强度树训练失败，请检查数据或参数")

        sum_shear = np.sum(self.weights_shear)
        sum_peel = np.sum(self.weights_peel)
        self.weights_shear = np.array(self.weights_shear) / (sum_shear + 1e-10)
        self.weights_peel = np.array(self.weights_peel) / (sum_peel + 1e-10)

    def predict(self, X):
        shear_preds = []
        for i, (tree, feature_indices) in enumerate(self.trees_shear):
            pred = tree.predict(X[:, feature_indices])
            shear_preds.append(pred * self.weights_shear[i])
        pred_shear = np.sum(np.array(shear_preds), axis=0).flatten()

        peel_preds = []
        for i, (tree, feature_indices) in enumerate(self.trees_peel):
            pred = tree.predict(X[:, feature_indices])
            peel_preds.append(pred * self.weights_peel[i])
        pred_peel = np.sum(np.array(peel_preds), axis=0).flatten()

        pred_shear = np.nan_to_num(pred_shear, nan=0.0, posinf=1e20, neginf=-1e20)
        pred_peel = np.nan_to_num(pred_peel, nan=0.0, posinf=1e20, neginf=-1e20)

        return pred_shear, pred_peel


# ----------------------
# 3. PSO优化器（小样本适配）
# ----------------------
class PSOOptimizer:
    def __init__(self, X_train, y_shear_train, y_peel_train, X_val, y_shear_val, y_peel_val, param_bounds,
                 n_particles=6, max_iter=10):  # 粒子数8→6，迭代15→10，加速小样本训练
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

            # 小样本参数约束：树数20→10，剪枝阈值1→2，预测试占比0.15→0.05
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

            if np.isnan(fitness) or np.isinf(fitness):
                return -1e9
            return fitness
        except Exception as e:
            print(f"警告：参数组合 {params.round(2)} 训练失败，适应度设为-1e9。错误：{str(e)[:150]}")
            return -1e9

    def _initialize_particles(self):
        n_params = len(self.param_bounds)
        particles = np.zeros((self.n_particles, n_params))
        for i in range(n_params):
            particles[:, i] = np.random.uniform(self.param_bounds[i][0], self.param_bounds[i][1], self.n_particles)
        velocities = np.random.uniform(-0.5, 0.5, (self.n_particles, n_params))  # 速度从±1→±0.5，避免参数震荡
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


# ----------------------
# 4. PSO优化加权随机森林主类（核心修改：数据拆分策略）
# ----------------------
class PSOWRFRegressor:
    def __init__(self, param_bounds=None, pso_kwargs=None):
        # 小样本参数边界：树数20→10，剪枝阈值0→2，预测试占比0.1→0.05
        self.param_bounds = param_bounds or [
            (10, 50),  # 决策树棵数（原20-80→10-50）
            (2, 5),  # 剪枝阈值（原0-10→2-5）
            (0.05, 0.15),  # 预测试样本占比（原0.1-0.2→0.05-0.15）
            (2, 5)  # 随机特征数（保持不变）
        ]
        self.pso_kwargs = pso_kwargs or {"n_particles": 6, "max_iter": 10}
        self.best_params = None
        self.wrf = None

    def fit(self, X, y_shear, y_peel):
        if X.ndim != 2 or y_shear.ndim != 1 or y_peel.ndim != 1:
            raise ValueError("X必须是2维数组，y_shear和y_peel必须是1维数组")

        # 核心修改：小样本拆分策略
        total_samples = len(X)
        print(f"总样本数：{total_samples}")

        # 根据总样本数动态调整验证集占比
        if total_samples <= 20:
            test_size = 0.1  # 20个样本→验证集2个（原0.2→0.1）
        elif total_samples <= 50:
            test_size = 0.15  # 50个样本→验证集7-8个
        else:
            test_size = 0.2

        try:
            X_train, X_val, y_shear_train, y_shear_val, y_peel_train, y_peel_val = train_test_split(
                X, y_shear, y_peel, test_size=test_size, random_state=42, shuffle=True
            )
            # 放宽样本数限制：训练集≥10，验证集≥2（原训练集≥5，验证集≥5）
            if len(X_train) < 10 or len(X_val) < 2:
                raise ValueError(f"拆分后样本数不足（训练集 {len(X_train)} 个，验证集 {len(X_val)} 个），请增加数据量")

            print(f"拆分后：训练集 {len(X_train)} 个，验证集 {len(X_val)} 个")
        except Exception as e:
            raise ValueError(f"数据拆分失败：{str(e)}")

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
        print(f"决策树棵数（L）：{n_estimators}")
        print(f"剪枝阈值（ε）：{prune_threshold}")
        print(f"预测试样本占比（X_ratio）：{val_split_ratio:.2f}")
        print(f"随机特征数（m）：{max_features}")

    def predict(self, X):
        return self.wrf.predict(X)


# ----------------------
# 5. 主函数（不清洗原始数据）
# ----------------------
def main():
    # 1. 读取Excel数据（不清洗）
    excel_path = "模拟数据.xlsx"
    try:
        data = pd.read_excel(excel_path, sheet_name=0)
        print("Excel数据读取成功！数据形状：", data.shape)
        print("原始数据前5行：")
        print(data.head())
    except Exception as e:
        print(f"Excel读取失败：{e}")
        return

    # 2. 提取原始数据（不清洗）
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

    # 3. 训练模型（小样本适配）
    print("\n=== PSO优化加权随机森林训练中 ===")
    try:
        psowrf = PSOWRFRegressor(
            param_bounds=[(10, 50), (2, 5), (0.05, 0.15), (2, 5)],
            pso_kwargs={"n_particles": 6, "max_iter": 10}
        )
        psowrf.fit(X, y_shear, y_peel)
    except Exception as e:
        print(f"模型训练失败：{e}")
        return

    # 4. 模型评估
    psowrf_shear_pred, psowrf_peel_pred = psowrf.predict(X)
    y_shear_eval = np.nan_to_num(y_shear, nan=np.nanmean(y_shear) if not np.isnan(y_shear).all() else 0.0,
                                 posinf=np.percentile(y_shear[np.isfinite(y_shear)], 99) if np.isfinite(
                                     y_shear).any() else 0.0,
                                 neginf=np.percentile(y_shear[np.isfinite(y_shear)], 1) if np.isfinite(
                                     y_shear).any() else 0.0)
    y_peel_eval = np.nan_to_num(y_peel, nan=np.nanmean(y_peel) if not np.isnan(y_peel).all() else 0.0,
                                posinf=np.percentile(y_peel[np.isfinite(y_peel)], 99) if np.isfinite(
                                    y_peel).any() else 0.0,
                                neginf=np.percentile(y_peel[np.isfinite(y_peel)], 1) if np.isfinite(
                                    y_peel).any() else 0.0)

    print(f"\n剪切强度R²：{r2_score(y_shear_eval, psowrf_shear_pred):.3f}")
    print(f"扯离强度R²：{r2_score(y_peel_eval, psowrf_peel_pred):.3f}")

    # 5. 最优温度搜索
    print("\n=== 最优温度条件搜索中 ===")
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

    print("\n=== PSO优化加权随机森林 - 最优温度条件 ===")
    print(f"环境温度：{best_temp_params[0]:.1f}℃")
    print(f"固化温度：{best_temp_params[1]:.1f}℃")
    print(f"推进剂浇筑温度：{best_temp_params[2]:.1f}℃")
    print(f"最优剪切强度：{best_shear:.3f}MPa")
    print(f"最优扯离强度：{best_peel:.3f}kN/m")
    print(f"综合强度（剪切+扯离）：{best_combined:.3f}")


if __name__ == "__main__":
    main()