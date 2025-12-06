import numpy as np
import matplotlib.pyplot as plt

# 彻底禁用中文，使用纯英文字体（避免所有中文字体警告）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------1. 实验曲线（不变）----------------------
def experiment_curve(epsilon):
    sigma = np.zeros_like(epsilon)
    # 线性段（0-9%）：斜率≈2.89 MPa（0.26/0.09）
    mask1 = epsilon < 0.09
    sigma[mask1] = (0.26 / 0.09) * epsilon[mask1]
    # 非线性上升段（9%-14%）：从0.26→0.39 MPa
    mask2 = (epsilon >= 0.09) & (epsilon < 0.14)
    sigma[mask2] = 0.26 + 0.13 * ((epsilon[mask2] - 0.09) / 0.05) ** 2
    # 下降段（14%-20%）：从0.39→0 MPa
    mask3 = epsilon >= 0.14
    sigma[mask3] = 0.39 - (0.39 / 0.06) * (epsilon[mask3] - 0.14)
    return sigma


epsilon_exp = np.linspace(0, 0.2, 200)  # 加密数据点
sigma_exp = experiment_curve(epsilon_exp)


# ----------------------2. 双线性模型（强物理约束）----------------------
def bilinear_model(epsilon, E, sigma_max, G):
    """双线性内聚力模型（强制物理合理性）"""
    # 参数硬约束（符合工程常识）
    E = np.clip(E, 2.0, 4.0)  # 弹性模量：2-4 MPa（匹配实验线性段斜率）
    sigma_max = np.clip(sigma_max, 0.35, 0.45)  # 峰值应力：0.35-0.45 MPa（实验峰值0.39）
    G = np.clip(G, 0.03, 0.05)  # 断裂能：0.03-0.05 kJ/m²（实验曲线面积≈0.039）

    # 模型逻辑约束（确保“上升→下降→失效”完整）
    delta_m_e = sigma_max / E  # 损伤起始位移（应力达峰值）
    delta_m_f = 2 * G / sigma_max  # 失效位移（断裂能=1/2*sigma_max*delta_m_f）
    delta_m_f = max(delta_m_f, delta_m_e + 0.02)  # 失效位移必须比起始位移大0.02
    delta_m_e = min(delta_m_e, 0.12)  # 损伤起始位移不超过0.12（峰值附近应变）

    delta_m = epsilon
    sigma_sim = np.zeros_like(delta_m)

    # 线性上升段（未损伤）
    mask1 = delta_m < delta_m_e
    sigma_sim[mask1] = E * delta_m[mask1]
    # 线性下降段（损伤演化）
    mask2 = (delta_m >= delta_m_e) & (delta_m <= delta_m_f)
    d = (delta_m[mask2] - delta_m_e) / (delta_m_f - delta_m_e)  # 损伤系数0→1
    sigma_sim[mask2] = (1 - d) * sigma_max
    # 完全失效（应力=0）
    mask3 = delta_m > delta_m_f
    sigma_sim[mask3] = 0

    return sigma_sim


# ----------------------3. 目标函数（加权拟合，重点关注关键段）----------------------
def objective_function(params):
    E, sigma_max, G = params
    sigma_sim = bilinear_model(epsilon_exp, E, sigma_max, G)

    # 给关键段加权重（提升拟合优先级）
    weights = np.ones_like(epsilon_exp)
    # 1. 峰值附近（9%-14%应变）：权重×3（重点拟合峰值）
    peak_mask = (epsilon_exp >= 0.09) & (epsilon_exp < 0.14)
    weights[peak_mask] = 3.0
    # 2. 下降段（14%-20%应变）：权重×2（重点拟合下降趋势）
    drop_mask = epsilon_exp >= 0.14
    weights[drop_mask] = 2.0

    # 加权均方误差（WMSE）：关键段误差影响更大
    weighted_mse = np.mean(weights * (sigma_sim - sigma_exp) ** 2)
    return weighted_mse


# ----------------------4. Hook-Jeeves算法（多初始点+动态步长）----------------------
def hook_jeeves_multi_start(fun, x0_list, step=0.02, tol=1e-5, max_iter=300, step_shrink=0.4):
    """多初始点Hook-Jeeves：避免局部最优，选择最优结果"""
    best_result = None
    min_f = float('inf')

    for idx, x0 in enumerate(x0_list):
        print(f"Running {idx + 1}/{len(x0_list)} initial point: {x0}")
        x_current = np.array(x0, dtype=np.float64)
        f_current = fun(x_current)
        history = [x_current.copy()]
        step_current = step

        for iter_count in range(max_iter):
            # 1. 探测搜索（逐维试探）
            x_probe = x_current.copy()
            improved = False
            for i in range(len(x_current)):
                # 正方向
                x_temp = x_probe.copy()
                x_temp[i] += step_current
                f_temp = fun(x_temp)
                if f_temp < f_current:
                    f_current = f_temp
                    x_probe = x_temp
                    improved = True
                else:
                    # 负方向
                    x_temp[i] -= 2 * step_current
                    f_temp = fun(x_temp)
                    if f_temp < f_current:
                        f_current = f_temp
                        x_probe = x_temp
                        improved = True

            # 2. 模式搜索（加速）
            if iter_count > 0:
                x_pattern = 2 * x_probe - history[-2]
                f_pattern = fun(x_pattern)
                if f_pattern < f_current:
                    x_current = x_pattern
                    f_current = f_pattern
                    improved = True
                else:
                    x_current = x_probe
            else:
                x_current = x_probe

            # 3. 动态步长：无改进则收缩，有改进则保持
            if not improved:
                step_current *= step_shrink
                if step_current < 1e-7:
                    break

            history.append(x_current.copy())

            # 4. 收敛判断
            if len(history) > 1 and np.linalg.norm(history[-1] - history[-2]) < tol:
                break

        # 记录最优初始点的结果
        if f_current < min_f:
            min_f = f_current
            best_result = (x_current, f_current, np.array(history))

    return best_result[0], best_result[1], best_result[2]


# ----------------------5. 执行反演（多初始点+精准参数范围）----------------------
# 多初始点：覆盖参数合理范围，避免局部最优
x0_list = [
    [2.8, 0.38, 0.038],  # 接近实验特征的初始点（最优候选）
    [2.5, 0.39, 0.039],
    [3.2, 0.40, 0.040],
    [2.7, 0.37, 0.037],
    [3.1, 0.41, 0.041]
]

# 运行多初始点优化
best_params, min_error, history = hook_jeeves_multi_start(objective_function, x0_list)
sigma_sim_best = bilinear_model(epsilon_exp, *best_params)


# ----------------------6. 误差量化----------------------
def calculate_errors(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return mse, rmse, mae, r2


mse, rmse, mae, r2 = calculate_errors(sigma_exp, sigma_sim_best)

# ----------------------7. 结果输出与可视化（纯英文，无字体警告）----------------------
print("\n" + "=" * 60)
print("Inversion Result Summary (R² Optimized Version)")
print("=" * 60)
print(f"Optimal Parameters:")
print(f"  Elastic Modulus E = {best_params[0]:.3f} MPa")
print(f"  Peak Stress sigma_max = {best_params[1]:.3f} MPa")
print(f"  Fracture Energy G = {best_params[2]:.4f} kJ/m²")
print(f"\nError Metrics (Key):")
print(f"  Root Mean Square Error (RMSE) = {rmse:.4f} MPa")
print(f"  Coefficient of Determination (R²) = {r2:.4f}")  # 预期≥0.96
print(f"  Mean Absolute Error (MAE) = {mae:.4f} MPa")
print("=" * 60)

# 图1：实验vs模拟曲线（突出关键段）
plt.figure(figsize=(10, 6))
plt.plot(epsilon_exp, sigma_exp, 'r-', label='Experiment Curve', linewidth=2.5)
plt.plot(epsilon_exp, sigma_sim_best, 'b--', label='Inverted Simulation Curve', linewidth=2.5, alpha=0.9)
# 标注关键段（英文，避免中文）
plt.axvspan(0.09, 0.14, alpha=0.1, color='orange', label='Peak Region (Weight×3)')
plt.axvspan(0.14, 0.2, alpha=0.1, color='green', label='Drop Region (Weight×2)')
# 误差文本
error_text = f'RMSE = {rmse:.4f} MPa\nR² = {r2:.4f}\nMAE = {mae:.4f} MPa'
plt.text(0.15, 0.32, error_text, fontsize=11, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
plt.xlabel('Strain', fontsize=12)
plt.ylabel('Stress (MPa)', fontsize=12)
plt.title('Hook-Jeeves Inversion: Experiment vs Simulation', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

# 图2：参数收敛过程
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(history[:, 0], 'o-', color='blue', markersize=3, linewidth=1.5, label=f'E = {best_params[0]:.3f} MPa')
plt.xlabel('Iteration', fontsize=11)
plt.ylabel('Elastic Modulus E (MPa)', fontsize=11)
plt.legend()
plt.grid(alpha=0.3)
plt.subplot(132)
plt.plot(history[:, 1], 's-', color='orange', markersize=3, linewidth=1.5,
         label=f'sigma_max = {best_params[1]:.3f} MPa')
plt.xlabel('Iteration', fontsize=11)
plt.ylabel('Peak Stress (MPa)', fontsize=11)
plt.legend()
plt.grid(alpha=0.3)
plt.subplot(133)
plt.plot(history[:, 2], '^-', color='green', markersize=3, linewidth=1.5, label=f'G = {best_params[2]:.4f} kJ/m²')
plt.xlabel('Iteration', fontsize=11)
plt.ylabel('Fracture Energy G (kJ/m²)', fontsize=11)
plt.legend()
plt.grid(alpha=0.3)
plt.suptitle('Parameter Convergence Process', fontsize=14)
plt.tight_layout()
plt.show()

# 图3：残差分布
plt.figure(figsize=(8, 5))
residuals = sigma_exp - sigma_sim_best
plt.hist(residuals, bins=30, alpha=0.7, color='lightcoral', edgecolor='black',
         label=f'Residual Mean = {np.mean(residuals):.4f}')
plt.axvline(x=0, color='darkred', linestyle='--', linewidth=2, label='Zero Error')
plt.xlabel('Residual (Experiment - Simulation) (MPa)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Residual Distribution Histogram', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.show()