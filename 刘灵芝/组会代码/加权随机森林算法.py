import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import mode


# ----------------------
# C4.5决策树类（回归版，复用文档1.2节核心逻辑，同普通随机森林）
# ----------------------
class C45DecisionTree:
    def __init__(self, prune_threshold=5):
        self.root = None  # 决策树根节点（存储树结构）
        self.prune_threshold = prune_threshold  # 剪枝阈值ε（文档1.2节，样本数小于ε时停止分裂，避免过拟合）

    # 计算熵（文档1.1节公式1，Entropy(S)，衡量样本集合纯度，回归任务中用于特征分裂依据）
    def _entropy(self, y):
        # 回归任务中对连续值y离散化处理（按unique值分组，模拟分类任务的“类别”）
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)  # 各组样本占比p_i
        return -np.sum(probs * np.log2(probs + 1e-10))  # 熵计算公式，+1e-10避免log(0)

    # 计算信息增益（文档1.1节公式2，Gain(S,A)，衡量特征A对分裂的贡献）
    def _information_gain(self, X, y, feature_idx):
        base_entropy = self._entropy(y)  # 原始样本集合的熵
        feature_values = np.unique(X[:, feature_idx])  # 当前特征的所有取值
        weighted_entropy = 0.0  # 分裂后的加权熵（按子样本集大小加权）

        for val in feature_values:
            mask = X[:, feature_idx] == val  # 筛选特征值为val的子样本集
            subset_y = y[mask]
            # 累加子样本集的熵（按子样本集占比加权）
            weighted_entropy += (len(subset_y) / len(y)) * self._entropy(subset_y)

        return base_entropy - weighted_entropy  # 信息增益=原始熵-加权熵

    # 计算信息增益率（文档1.2节C4.5核心改进，解决信息增益偏向多值特征的问题）
    def _gain_ratio(self, X, y, feature_idx):
        gain = self._information_gain(X, y, feature_idx)
        feature_values = np.unique(X[:, feature_idx])
        # 关键修复：用 np.array 包裹，确保是 ndarray
        probs = np.array([len(X[X[:, feature_idx] == val]) / len(X) for val in feature_values])
        split_info = -np.sum(probs * np.log2(probs + 1e-10))
        return gain / (split_info + 1e-10)  # 增益率=信息增益/分裂信息，避免除以0

    # 寻找最优分裂特征（文档1.2节C4.5逻辑，选择增益率最大的特征）
    def _best_split(self, X, y, feature_indices):
        best_gain_ratio = -np.inf  # 初始化最优增益率为负无穷
        best_feature = None  # 初始化最优分裂特征索引

        for idx in feature_indices:
            current_gain_ratio = self._gain_ratio(X, y, idx)
            if current_gain_ratio > best_gain_ratio:
                best_gain_ratio = current_gain_ratio
                best_feature = idx

        return best_feature  # 返回最优分裂特征（无有效特征时返回None）

    # 递归构建决策树（含前剪枝，文档1.2节C4.5树生成流程）
    def _build_tree(self, X, y, feature_indices, depth=0):
        # 终止条件1：节点样本数小于剪枝阈值ε，转为叶节点（前剪枝）
        if len(y) <= self.prune_threshold:
            return {"type": "leaf", "value": np.mean(y)}  # 回归任务叶节点输出样本均值
        # 终止条件2：样本值差异极小（趋于一致），无需分裂
        if np.max(y) - np.min(y) < 1e-6:
            return {"type": "leaf", "value": y[0]}
        # 终止条件3：无剩余特征可分裂，转为叶节点
        if len(feature_indices) == 0:
            return {"type": "leaf", "value": np.mean(y)}

        # 选择最优分裂特征
        best_feature = self._best_split(X, y, feature_indices)
        # 若无有效分裂特征（所有特征增益率无效），转为叶节点
        if best_feature is None:
            return {"type": "leaf", "value": np.mean(y)}

        # 构建分裂节点，递归生成子树
        tree = {"type": "node", "feature": best_feature, "children": {}}
        feature_values = np.unique(X[:, best_feature])  # 最优特征的所有取值
        remaining_features = [f for f in feature_indices if f != best_feature]  # 移除已用特征

        for val in feature_values:
            mask = X[:, best_feature] == val
            # 为每个特征取值生成子树
            tree["children"][val] = self._build_tree(
                X[mask], y[mask], remaining_features, depth + 1
            )

        return tree

    # 训练决策树（文档1.2节，从根节点开始构建树）
    def fit(self, X, y):
        feature_indices = list(range(X.shape[1]))  # 所有特征的索引列表
        self.root = self._build_tree(X, y, feature_indices)  # 递归构建树并赋值给根节点

    # 单个样本预测（递归遍历决策树）
    # 单个样本预测（递归遍历决策树）
    def _predict_sample(self, x, node):
        if node["type"] == "leaf":  # 到达叶节点，返回叶节点值（回归结果）
            return node["value"]

        feature_val = x[node["feature"]]  # 获取当前样本在分裂特征上的取值
        # 容错处理：若特征值未在训练集中出现，递归收集所有子节点的叶节点值并取均值
        if feature_val not in node["children"]:
            # 递归函数：收集某个节点下所有叶节点的value
            def collect_leaf_values(current_node):
                leaf_values = []
                if current_node["type"] == "leaf":
                    leaf_values.append(current_node["value"])
                else:
                    # 遍历当前节点的所有子节点，递归收集
                    for child in current_node["children"].values():
                        leaf_values.extend(collect_leaf_values(child))
                return leaf_values

            # 收集当前节点下所有叶节点的value，取均值作为预测结果
            all_leaf_values = collect_leaf_values(node)
            return np.mean(all_leaf_values) if all_leaf_values else 0.0  # 空列表容错（返回0.0）

        # 递归进入子节点继续预测
        return self._predict_sample(x, node["children"][feature_val])


        # 批量预测（对所有测试样本执行预测）
    def predict(self, X):
        # 确保返回的是二维数组 (n_samples, 1)，统一格式
        predictions = np.array([self._predict_sample(x, self.root) for x in X]).reshape(-1, 1)
        return predictions


# ----------------------
# 加权随机森林回归类（双目标，文档2.2节核心改进：决策树加权投票）
# 说明：解决传统随机森林“等权投票”中劣质树干扰问题，按树性能赋予权重
# ----------------------
class WeightedRandomForestRegressor:
    def __init__(self, n_estimators=50, prune_threshold=5, max_features="sqrt", val_split_ratio=0.1):
        self.n_estimators = n_estimators  # 决策树棵数L（文档2.1节）
        self.prune_threshold = prune_threshold  # 剪枝阈值ε（文档1.2节）
        self.max_features = max_features  # 每棵树随机选择的特征数m（文档2.1节）
        self.val_split_ratio = val_split_ratio  # 预测试样本占比（文档2.2节，用于计算树权重）
        self.trees_shear = []  # 剪切强度预测树及对应特征索引
        self.trees_peel = []  # 扯离强度预测树及对应特征索引
        self.weights_shear = []  # 剪切强度树的权重（文档2.2节公式5）
        self.weights_peel = []  # 扯离强度树的权重（文档2.2节公式5）

    # Bootstrap抽样+预测试样本拆分（文档2.2节核心改进：保证权重计算的公平性）
    def _bootstrap_with_val(self, X, y_shear, y_peel):
        n_samples = X.shape[0]
        # 1. Bootstrap抽样生成训练子集（用于训练单棵决策树）
        train_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_train = X[train_indices]
        y_shear_train = y_shear[train_indices]
        y_peel_train = y_peel[train_indices]

        # 2. 拆分预测试样本（从原始数据中随机选择，非袋外样本，文档2.2节公平性设计）
        val_size = int(n_samples * self.val_split_ratio)  # 预测试样本数X（文档2.2节）
        val_indices = np.random.choice(n_samples, size=val_size, replace=False)
        X_val = X[val_indices]
        y_shear_val = y_shear[val_indices]
        y_peel_val = y_peel[val_indices]

        return X_train, y_shear_train, y_peel_train, X_val, y_shear_val, y_peel_val

    # 计算决策树权重（文档2.2节公式5：权重=预测试样本正确率，此处适配回归用R²）
    def _calculate_weight(self, y_true, y_pred):
        # 回归任务中用R²替代分类正确率（R²越接近1，树性能越好，权重越大）
        r2 = r2_score(y_true, y_pred)
        return max(r2, 0.1)  # 权重下限0.1，避免劣质树权重为0导致信息丢失

    # 训练加权随机森林（文档2.2节流程：Bootstrap+预测试+权重计算）
    def fit(self, X, y_shear, y_peel):
        self.trees_shear = []
        self.trees_peel = []
        self.weights_shear = []
        self.weights_peel = []
        n_features = X.shape[1]

        # 确定每棵树随机选择的特征数m（文档2.1节默认m=√M）
        self.m = int(np.sqrt(n_features)) if self.max_features == "sqrt" else self.max_features

        # 生成n_estimators棵决策树（双目标并行训练+权重计算）
        for _ in range(self.n_estimators):
            # 1. Bootstrap抽样+拆分预测试样本
            X_train, y_shear_train, y_peel_train, X_val, y_shear_val, y_peel_val = self._bootstrap_with_val(X, y_shear,
                                                                                                            y_peel)
            # 2. 随机选择m个特征
            feature_indices = np.random.choice(n_features, size=self.m, replace=False)

            # 3. 训练剪切强度树+计算权重
            tree_shear = C45DecisionTree(prune_threshold=self.prune_threshold)
            tree_shear.fit(X_train[:, feature_indices], y_shear_train)
            # 用预测试样本评估树性能（文档2.2节，计算权重的依据）
            y_shear_val_pred = tree_shear.predict(X_val[:, feature_indices])
            weight_shear = self._calculate_weight(y_shear_val, y_shear_val_pred)
            self.trees_shear.append((tree_shear, feature_indices))
            self.weights_shear.append(weight_shear)

            # 4. 训练扯离强度树+计算权重（同剪切强度逻辑）
            tree_peel = C45DecisionTree(prune_threshold=self.prune_threshold)
            tree_peel.fit(X_train[:, feature_indices], y_peel_train)
            y_peel_val_pred = tree_peel.predict(X_val[:, feature_indices])
            weight_peel = self._calculate_weight(y_peel_val, y_peel_val_pred)
            self.trees_peel.append((tree_peel, feature_indices))
            self.weights_peel.append(weight_peel)

        # 权重归一化（文档2.2节：确保权重和为1，避免权重过大的树过度主导）
        self.weights_shear = np.array(self.weights_shear) / np.sum(self.weights_shear)
        self.weights_peel = np.array(self.weights_peel) / np.sum(self.weights_peel)

    # 预测（文档2.2节公式6：加权均值平均，替代传统等权平均）
    def predict(self, X):
        # 1. 剪切强度加权预测
        shear_preds = []
        for i, (tree, feature_indices) in enumerate(self.trees_shear):
            pred = tree.predict(X[:, feature_indices])
            # 预测结果乘以对应树的权重（文档2.2节加权逻辑）
            shear_preds.append(pred * self.weights_shear[i])
        # 加权求和（权重已归一化，等价于加权均值）
        pred_shear = np.sum(np.array(shear_preds), axis=0)

        # 2. 扯离强度加权预测（同剪切强度逻辑）
        peel_preds = []
        for i, (tree, feature_indices) in enumerate(self.trees_peel):
            pred = tree.predict(X[:, feature_indices])
            peel_preds.append(pred * self.weights_peel[i])
        pred_peel = np.sum(np.array(peel_preds), axis=0)

        return pred_shear, pred_peel


# ----------------------
# 主函数（读取Excel+训练+评估+最优温度搜索，同普通随机森林工程化逻辑）
# ----------------------
def main():
    # 1. 读取Excel数据
    excel_path = "模拟数据.xlsx"
    try:
        data = pd.read_excel(excel_path, sheet_name=0)
        print("Excel数据读取成功！数据形状：", data.shape)
    except Exception as e:
        print(f"Excel读取失败：{e}")
        return

    # 2. 数据预处理
    feature_cols = ['环境温度(℃)', '环境湿度(%)', '固化温度(℃)', '固化时间(h)', '推进剂浇筑温度(℃)']
    target_shear_col = '剪切强度(MPa)'
    target_peel_col = '扯离强度(kN/m)'

    X = data[feature_cols].values
    y_shear = data[target_shear_col].values
    y_peel = data[target_peel_col].values

    # 3. 训练加权随机森林（参数参考文档3节：树数50，剪枝阈值5，预测试占比0.2）
    print("\n=== 加权随机森林训练中 ===")
    wrf = WeightedRandomForestRegressor(n_estimators=50, prune_threshold=5, val_split_ratio=0.2)
    wrf.fit(X, y_shear, y_peel)

    # 4. 模型评估（文档3节用R²衡量回归性能）
    wrf_shear_pred, wrf_peel_pred = wrf.predict(X)
    print(f"剪切强度R²：{r2_score(y_shear, wrf_shear_pred):.3f}")
    print(f"扯离强度R²：{r2_score(y_peel, wrf_peel_pred):.3f}")
    print(f"前5棵剪切强度树权重：{wrf.weights_shear[:5].round(3)}")  # 展示部分权重，验证加权逻辑

    # 5. 最优温度搜索（同普通随机森林，工程化目标）
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
                pred_shear, pred_peel = wrf.predict(input_data)

                # 强制转为 Python float，彻底解决格式化问题
                shear_val = pred_shear.item()
                peel_val = pred_peel.item()

                combined_strength = shear_val + peel_val

                if combined_strength > best_combined:
                    best_combined = combined_strength
                    best_shear = shear_val
                    best_peel = peel_val
                    best_temp_params = (env, cure, pour)

    # 输出最优结果
    print("\n=== 加权随机森林 - 最优温度条件 ===")
    print(f"环境温度：{best_temp_params[0]:.1f}℃")
    print(f"固化温度：{best_temp_params[1]:.1f}℃")
    print(f"推进剂浇筑温度：{best_temp_params[2]:.1f}℃")
    print(f"最优剪切强度：{best_shear:.3f}MPa")  # 现在是float，正常格式化
    print(f"最优扯离强度：{best_peel:.3f}kN/m")  # 现在是float，正常格式化
    print(f"综合强度（剪切+扯离）：{best_combined:.3f}")


if __name__ == "__main__":
    main()