import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_data(data_path, output_columns, n_samples, drop_columns=None):
    """
    数据预处理函数，返回分割后的训练集和测试集

    参数:
        data_path (str): 数据集路径
        output_columns (list): 目标变量Y的列名列表
        n_samples (int): 选择的样本数量
        drop_columns (list, optional): 该参数已忽略（原用于排除特征列，现使用指定列作为X特征），默认为空列表

    返回:
        X_train, X_test, Y_train, Y_test: 分割后的训练集和测试集
    """
    # 处理默认参数（drop_columns仅保留参数兼容，实际不使用）
    if drop_columns is None:
        drop_columns = []
    else:
        print("警告：drop_columns参数已忽略，当前使用指定列作为X特征")

    # 定义需要作为X特征的列名
    x_feature_columns = [
        "AmbientConditions.AmbientHumidity.U.Actual",
        "AmbientConditions.AmbientTemperature.U.Actual",
        "Machine1.RawMaterialFeederParameter.U.Actual",
        "Machine1.Zone1Temperature.C.Actual",
        "Machine1.Zone2Temperature.C.Actual",
        "Machine1.MaterialPressure.U.Actual",
        "Machine1.MaterialTemperature.U.Actual",
        "Machine2.RawMaterialFeederParameter.U.Actual",
        "Machine2.Zone1Temperature.C.Actual",
        "Machine2.Zone2Temperature.C.Actual",
        "Machine2.MaterialPressure.U.Actual",
        "Machine2.MaterialTemperature.U.Actual",
        "Machine3.RawMaterial.Property1",
        "Machine3.RawMaterial.Property2",
        "Machine3.RawMaterial.Property3"
    ]

    # 1. 加载数据集并清洗缺失值
    sampled_data = pd.read_csv(data_path)
    cleaned_data = sampled_data.dropna()  # 删除含缺失值的行
    print(f"清洗后总样本数：{cleaned_data.shape[0]}")

    # 2. 随机选择n个样本进入实验
    if n_samples > cleaned_data.shape[0]:
        selected_data = cleaned_data
        print(f"警告：n_samples({n_samples})大于清洗后样本数，已使用全部{cleaned_data.shape[0]}个样本")
    else:
        selected_data = cleaned_data.sample(n=n_samples, random_state=6)
    print(f"进入实验的样本数：{selected_data.shape[0]}")

    # 3. 确定X的特征列为指定列，Y为输入的目标列
    all_columns = selected_data.columns.tolist()

    # 检查数据集是否包含所有必要的X特征列
    missing_columns = [col for col in x_feature_columns if col not in all_columns]
    if missing_columns:
        raise ValueError(f"数据集缺少以下必要特征列：{missing_columns}")
    X_columns = x_feature_columns

    # 检查目标列是否包含在X特征列中（避免数据泄露）
    for y_col in output_columns:
        if y_col in X_columns:
            raise ValueError(f"目标列'{y_col}'位于X特征列中，会导致数据泄露，请调整目标列设置")

    # 分离X和Y
    X = selected_data[X_columns]
    Y = selected_data[output_columns].values.ravel()

    # 4. 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_columns)

    # 5. 分割训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled_df, Y,
        test_size=0.4,
        random_state=30
    )

    # 打印信息确认
    print(f"\nX的特征列为指定列，列名：\n{X_columns}\n")
    print(f"Y的目标列数量：{len(output_columns)}，列名：\n{output_columns}\n")
    print(f"训练集样本数：{X_train.shape[0]}，测试集样本数：{X_test.shape[0]}")
    print(f"（训练集+测试集）总样本数：{X_train.shape[0] + X_test.shape[0]}（与选中的样本数一致）")

    return X_train, X_test, Y_train, Y_test