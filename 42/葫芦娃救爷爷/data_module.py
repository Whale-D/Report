# data_module.py
import numpy as np
from 拿数据 import prepare_data
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.datasets import make_friedman1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def load_friedman_data(n, n_features=20, noise=0, random_state=42):
    """
    生成一个20维Friedman非线性回归数据集，并抽取前n条样本。

    参数:
        n: 使用的样本数量
        n_features: 特征维度，默认20
        noise: 噪声大小
        random_state: 随机种子（保证可复现）

    返回:
        X_train, X_test, Y_train, Y_test
    """

    # ============= 1. 生成 Friedman #1 非线性数据 =============
    X, Y = make_friedman1(
        n_samples=max(n, 200),       # 生成足够的样本，取前 n 条
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )

    feature_names = [f"Feature_{i}" for i in range(n_features)]

    print("Friedman #1 Nonlinear Dataset Information:")
    print(f"Total samples generated: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {feature_names}")
    print(f"Target range: {Y.min():.3f} - {Y.max():.3f}")
    print(f"Target mean: {Y.mean():.3f}\n")

    # ============= 2. 取前 n 条样本 =============
    X, Y = X[:n], Y[:n]

    # ============= 3. 特征标准化 =============
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ============= 4. 切分训练/测试 =============
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.3, random_state=random_state
    )

    print("Dataset ready:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")

    return X_train, X_test, Y_train, Y_test

def load_boston_data(n, excel_file=r'D:\python\42\葫芦娃救爷爷\boston_housing_data.xlsx', random_state=42):
    """
    可复现的按Y分布分层抽样，确保无论运行多少次结果都相同
    """

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # 创建独立且可复现的随机数生成器
    rng = np.random.default_rng(random_state)

    # 1. 读取数据
    df = pd.read_excel(excel_file, sheet_name='房价数据')

    # 自动识别 PRICE 列
    target_column = None
    for col in ['PRICE', 'MEDV', 'Price', 'price', 'medv', 'target']:
        if col in df.columns:
            target_column = col
            break
    if target_column is None:
        target_column = df.columns[-1]

    feature_cols = [c for c in df.columns if c not in ['样本ID', target_column]]

    X = df[feature_cols].values
    Y = df[target_column].values

    # ======================================================
    #  分层抽样（deterministic stratified sampling）
    # ======================================================

    num_bins = 10
    bins = np.linspace(Y.min(), Y.max(), num_bins + 1)
    y_bins = np.digitize(Y, bins)

    per_bin = max(1, n // num_bins)
    sampled_idx = []

    for b in range(1, num_bins + 1):
        idx = np.where(y_bins == b)[0]
        if len(idx) > 0:
            # 使用确定性的 rng 而不是 numpy 全局
            chosen = rng.choice(idx, size=min(per_bin, len(idx)), replace=False)
            sampled_idx.extend(chosen.tolist())

    # 如果不足 n，补齐到 n
    if len(sampled_idx) < n:
        remaining = np.setdiff1d(np.arange(len(Y)), sampled_idx)
        need = n - len(sampled_idx)
        fill = rng.choice(remaining, size=need, replace=False)
        sampled_idx.extend(fill.tolist())

    sampled_idx = np.array(sampled_idx[:n], dtype=int)

    # ---- 确保样本顺序一致（可复现）----
    sampled_idx = np.sort(sampled_idx)

    # 抽样后的数据
    X = X[sampled_idx]
    Y = Y[sampled_idx]

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分割训练/测试集 - 使用同一个随机种子确保复现
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled,
        Y,
        test_size=0.3,
        random_state=random_state,   # 保证可复现
        shuffle=True
    )

    print(f"使用基于Y分布的确定性抽样，成功选取 {n} 条样本（可复现）")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    return X_train, X_test, Y_train, Y_test

def load_house_data(n):
    # 1. 加载加州房价数据集
    housing = fetch_california_housing()
    X, Y = housing.data, housing.target
    feature_names = housing.feature_names

    print(f"Dataset Information:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {list(feature_names)}")
    print(f"Target range: ${Y.min():.2f}K - ${Y.max():.2f}K")
    print(f"Target mean: ${Y.mean():.2f}K")

    # 2. 数据预处理
    X, Y = X[:n], Y[:n]
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分割训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.3, random_state=42
    )
    return X_train, X_test, Y_train, Y_test


def load_air_quality_data(n):
    """
    加载并预处理空气质量数据集

    参数:
    n -- 要使用的样本数量

    返回:
    X_train, X_test, Y_train, Y_test -- 分割后的训练集和测试集
    """
    # 1. 加载空气质量数据集
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.csv"

    try:
        # 读取数据集，处理分隔符和编码问题
        df = pd.read_csv(url, sep=';', decimal=',', encoding='latin1')
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return None, None, None, None

    # 2. 数据预处理
    # 删除不必要的列
    df = df.drop(['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], axis=1, errors='ignore')

    # 处理缺失值（数据集中的缺失值表示为-200）
    df = df.replace(-200, np.nan)
    df = df.dropna()

    # 3. 分离特征和目标
    # 这里选择CO(GT)作为目标变量（一氧化碳浓度）
    target_col = 'CO(GT)'
    Y = df[target_col].values
    X = df.drop(target_col, axis=1).values
    feature_names = df.drop(target_col, axis=1).columns.tolist()

    print("\n空气质量数据集信息:")
    print(f"样本数量: {X.shape[0]}")
    print(f"特征数量: {X.shape[1]}")
    print(f"特征名称: {feature_names}")
    print(f"目标变量范围: {Y.min():.2f} - {Y.max():.2f}")
    print(f"目标变量均值: {Y.mean():.2f}")

    # 4. 限制样本数量
    if n > X.shape[0]:
        print(f"警告: 请求的样本数({n})超过可用样本数({X.shape[0]})，使用所有可用样本")
        n = X.shape[0]

    X, Y = X[:n], Y[:n]

    # 5. 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. 分割训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.3, random_state=42
    )
    print(X_train,Y_train)
    return X_train, X_test, Y_train, Y_test
def load_factory_data(
    data_path="filtered_sampled_data.csv",
    output_columns=None,
    n_samples=50,
    drop_columns=None
):
    """
    从工厂数据文件中读取数据，并做基础预处理（底层由 prepare_data 完成）

    返回：
    X_train, X_test, Y_train, Y_test
    """
    if output_columns is None:
        # 你现在脚本里用的那一列
        output_columns = ['Stage2.Output.Measurement14.U.Actual']
    if drop_columns is None:
        drop_columns = []

    X_train, X_test, Y_train, Y_test = prepare_data(
        data_path=data_path,
        output_columns=output_columns,
        n_samples=n_samples,
        drop_columns=drop_columns
    )

    return X_train, X_test, Y_train, Y_test


