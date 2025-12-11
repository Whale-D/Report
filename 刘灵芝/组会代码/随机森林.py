import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import mode


# ----------------------
# C4.5决策树类（回归版，文档1.2节核心算法改编）
# 说明：文档中C4.5为分类算法，此处适配回归任务，将叶节点输出从“类别众数”改为“样本均值”
# ----------------------
class C45DecisionTree:
    def __init__(self, prune_threshold=3):
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
# 普通随机森林回归类（双目标：剪切强度+扯离强度，文档2.1节随机森林核心逻辑）
# 说明：文档中随机森林为分类任务，此处适配回归任务，将“多数投票”改为“均值平均”
# ----------------------
class RandomForestRegressorCustom:
    def __init__(self, n_estimators=100, prune_threshold=5, max_features="sqrt"):
        self.n_estimators = n_estimators  # 决策树棵数L（文档2.1节，需经验选取）
        self.prune_threshold = prune_threshold  # 单棵树的剪枝阈值ε（文档1.2节）
        self.max_features = max_features  # 每棵树随机选择的特征数m（文档2.1节）
        self.trees_shear = []  # 存储预测“剪切强度”的决策树及对应特征索引
        self.trees_peel = []  # 存储预测“扯离强度”的决策树及对应特征索引

    # Bootstrap抽样（文档2.1节，有放回抽样生成子数据集，保证树的多样性）
    def _bootstrap_sample(self, X, y_shear, y_peel):
        n_samples = X.shape[0]
        # 有放回随机选择n个样本索引（允许重复，模拟Bagging思想）
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        # 返回抽样后的特征集、剪切强度标签集、扯离强度标签集
        return X[indices], y_shear[indices], y_peel[indices]

    # 训练随机森林（文档2.1节流程：Bootstrap抽样+随机特征选择+多棵树训练）
    def fit(self, X, y_shear, y_peel):
        self.trees_shear = []
        self.trees_peel = []
        n_features = X.shape[1]  # 总特征数M

        # 确定每棵树随机选择的特征数m（文档2.1节默认m=√M，保证特征多样性）
        if self.max_features == "sqrt":
            self.m = int(np.sqrt(n_features))
        else:
            self.m = self.max_features

        # 生成n_estimators棵决策树（双目标并行训练）
        for _ in range(self.n_estimators):
            # 1. Bootstrap抽样生成子数据集
            X_sample, y_shear_sample, y_peel_sample = self._bootstrap_sample(X, y_shear, y_peel)
            # 2. 随机选择m个特征（Random Subspace思想，文档2.1节）
            feature_indices = np.random.choice(n_features, size=self.m, replace=False)

            # 3. 训练“剪切强度”预测树
            tree_shear = C45DecisionTree(prune_threshold=self.prune_threshold)
            tree_shear.fit(X_sample[:, feature_indices], y_shear_sample)  # 仅用选中的m个特征训练
            self.trees_shear.append((tree_shear, feature_indices))  # 存储树和特征索引

            # 4. 训练“扯离强度”预测树（同剪切强度逻辑，双目标独立训练）
            tree_peel = C45DecisionTree(prune_threshold=self.prune_threshold)
            tree_peel.fit(X_sample[:, feature_indices], y_peel_sample)
            self.trees_peel.append((tree_peel, feature_indices))

    # 预测（文档2.1节回归适配：等权均值平均，替代分类的“多数投票”）
    def predict(self, X):
        # 1. 剪切强度预测：所有树预测结果的均值
        shear_preds = []
        for tree, feature_indices in self.trees_shear:
            # 每棵树仅用训练时选中的特征进行预测（返回 (n_samples,1)）
            pred = tree.predict(X[:, feature_indices])
            shear_preds.append(pred)
        # 等权平均：按列取均值，返回 (n_samples,)
        pred_shear = np.mean(np.array(shear_preds), axis=0).flatten()

        # 2. 扯离强度预测：同剪切强度逻辑
        peel_preds = []
        for tree, feature_indices in self.trees_peel:
            pred = tree.predict(X[:, feature_indices])
            peel_preds.append(pred)
        pred_peel = np.mean(np.array(peel_preds), axis=0).flatten()

        return pred_shear, pred_peel  # 返回双目标预测结果


# ----------------------
# 主函数（读取Excel数据+模型训练+评估+最优温度搜索，工程化适配）
# ----------------------
def main():
    # 1. 读取Excel数据（替换为实际文件路径，适配工程中数据存储场景）
    excel_path = "模拟数据.xlsx"
    try:
        # 读取Excel第一个工作表（假设数据无缺失，列名与代码一致）
        data = pd.read_excel(excel_path, sheet_name=0)
        print("Excel数据读取成功！数据形状：", data.shape)
    except Exception as e:
        print(f"Excel读取失败：{e}")
        print("请检查文件路径是否正确，或列名是否与代码匹配！")
        return

    # 2. 数据预处理（特征与目标变量分离，对应文档中“影响因素”与“性能指标”）
    # 特征列：环境温度、环境湿度、固化温度、固化时间、推进剂浇筑温度（影响界面脱粘的核心因素）
    feature_cols = ['环境温度(℃)', '环境湿度(%)', '固化温度(℃)', '固化时间(h)', '推进剂浇筑温度(℃)']
    # 目标列：剪切强度、扯离强度（评估界面脱粘质量的关键指标）
    target_shear_col = '剪切强度(MPa)'
    target_peel_col = '扯离强度(kN/m)'

    # 提取特征矩阵和目标向量（转为numpy数组便于模型计算）
    X = data[feature_cols].values
    y_shear = data[target_shear_col].values
    y_peel = data[target_peel_col].values

    # 3. 训练普通随机森林（参数参考文档3节实验：决策树棵数50，剪枝阈值5）
    print("\n=== 普通随机森林训练中 ===")
    rf = RandomForestRegressorCustom(n_estimators=50, prune_threshold=5)
    rf.fit(X, y_shear, y_peel)  # 双目标同时训练

    # 4. 模型评估（文档3节用R²衡量回归性能，R²越接近1表示拟合效果越好）
    rf_shear_pred, rf_peel_pred = rf.predict(X)
    print(f"剪切强度R²：{r2_score(y_shear, rf_shear_pred):.3f}")  # 剪切强度拟合优度
    print(f"扯离强度R²：{r2_score(y_peel, rf_peel_pred):.3f}")  # 扯离强度拟合优度

    # 5. 最优温度条件搜索（工程需求：寻找使双目标强度最优的温度组合）
    print("\n=== 最优温度条件搜索中 ===")
    # 生成温度候选范围（基于工程经验：环境温度20-35℃，固化温度55-70℃，浇筑温度50-65℃）
    env_temps = np.linspace(20, 35, 16)  # 环境温度：20-35℃，步长1℃（16个候选值）
    cure_temps = np.linspace(55, 70, 16)  # 固化温度：55-70℃，步长1℃
    pour_temps = np.linspace(50, 65, 16)  # 推进剂浇筑温度：50-65℃，步长1℃

    # 固定非温度参数（参考数据均值：湿度50%，固化时间4.5h）
    fixed_humidity = 50
    fixed_cure_time = 4.5

    # 初始化最优结果变量
    best_combined = 0  # 综合强度（剪切+扯离，作为优化目标）
    best_temp_params = None  # 最优温度组合（环境温度、固化温度、浇筑温度）
    best_shear = 0  # 最优剪切强度
    best_peel = 0  # 最优扯离强度

    # 遍历所有温度组合（暴力搜索，确保覆盖所有候选范围）
    for env in env_temps:
        for cure in cure_temps:
            for pour in pour_temps:
                # 构造单一样本输入（特征顺序与feature_cols一致）
                input_data = np.array([[env, fixed_humidity, cure, fixed_cure_time, pour]])
                # 预测当前温度组合下的双目标强度
                pred_shear, pred_peel = rf.predict(input_data)
                # 计算综合强度（简单求和，可根据工程权重调整）
                combined_strength = pred_shear[0] + pred_peel[0]

                # 更新最优结果（综合强度更大时替换）
                if combined_strength > best_combined:
                    best_combined = combined_strength
                    best_shear = pred_shear[0]
                    best_peel = pred_peel[0]
                    best_temp_params = (env, cure, pour)

    # 输出最优温度条件（工程化结果展示，便于实际生产参考）
    print("\n=== 普通随机森林 - 最优温度条件 ===")
    print(f"环境温度：{best_temp_params[0]:.1f}℃")
    print(f"固化温度：{best_temp_params[1]:.1f}℃")
    print(f"推进剂浇筑温度：{best_temp_params[2]:.1f}℃")
    print(f"最优剪切强度：{best_shear:.3f}MPa")
    print(f"最优扯离强度：{best_peel:.3f}kN/m")
    print(f"综合强度（剪切+扯离）：{best_combined:.3f}")


if __name__ == "__main__":
    main()