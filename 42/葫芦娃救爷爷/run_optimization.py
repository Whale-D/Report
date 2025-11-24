# run_optimization.py
from data_module import load_factory_data,load_house_data,load_air_quality_data,load_boston_data
from sparse_gp_model import build_sparse_gp_model
from bo_ucb import suggest_next_x_ucb


def main():
    # 1. 读取数据（根据需要改参数）
    X_train, X_test, Y_train, Y_test = load_boston_data(
        n=50
    )

    print(f"[数据] X_train 形状: {X_train.shape}, Y_train 形状: {Y_train.shape}")

    # 2. 训练稀疏高斯过程模型
    model_dict = build_sparse_gp_model(
        X_train=X_train,
        Y_train=Y_train,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    scaler = model_dict["scaler"]
    gp_for_std = model_dict["gp_for_std"]

    # 3. 使用 贝叶斯优化 + UCB 推荐下一组 X
    #   - 如果现在处于“探索阶段”，用 mode="ucb"
    #   - 如果已经大致找到好区域，希望更稳定，可以改成 mode="robust"
    next_x, next_mu, next_sigma = suggest_next_x_ucb(
        gaussian_model=gp_for_std,
        scaler=scaler,
        X_train_original=X_train,
        n_candidates=2000,
        kappa=2.0,
        mode="ucb",        # 或 "robust"
        # max_sigma=0.1,   # 若想强制方差小，可打开这一行调整阈值
        random_state=42
    )

    # 4. 输出结果
    print("\n========== 推荐的下一组工艺参数 X* ==========")
    print(next_x)
    print("===========================================")
    print(f"模型在该点预测的 Y 均值: {next_mu:.4f}")
    print(f"模型在该点预测的 Y 标准差: {next_sigma:.4f}")
    print("请在工厂用这组 X 做一次试验，测得真实 Y 后再加入数据继续迭代。")


if __name__ == "__main__":
    main()
