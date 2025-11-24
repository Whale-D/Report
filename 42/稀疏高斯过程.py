import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.stats as stats
import warnings
from 拿数据 import prepare_data

warnings.filterwarnings('ignore')

# 设置图形样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# # 1. 加载加州房价数据集
# housing = fetch_california_housing()
# X, Y = housing.data, housing.target
# feature_names = housing.feature_names
#
# # 2. 数据预处理 - 只取前50个样本保持简单
# X = X[:50]
# Y = Y[:50]
#
# # 特征标准化
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 分割训练集和测试集
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_scaled, Y, test_size=0.3, random_state=42
# )

from 葫芦娃救爷爷.data_module import load_boston_data,load_friedman_data
# X_train, X_test, Y_train, Y_test = load_boston_data(
#         n=50
#     )# 3. 定义管道和参数网格
X_train, X_test, Y_train, Y_test = load_friedman_data(
         n=50
     )
print("\nSetting up Grid Search for Sparse Gaussian Process Regression...")

# 创建管道 - 移除了多项式特征扩展
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 标准化
    ('gaussian', GaussianProcessRegressor())  # 高斯过程回归
])

# 定义参数网格 - 专注于稀疏核函数比较
# 使用稀疏高斯过程核函数配置
kernels = [
    # 稀疏RBF核 - 通过固定长度尺度和较小方差实现稀疏性
    C(0.1, (1e-3, 1e1)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e1)),

    # 稀疏Matern核 - 使用较大的nu值和固定长度尺度
    C(0.1, (1e-3, 1e1)) * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e1), nu=1.5),

    # 稀疏有理二次核 - 使用较大的alpha值实现稀疏性
    C(0.1, (1e-3, 1e1)) * RationalQuadratic(length_scale=0.5, alpha=10, alpha_bounds=(1, 100)),

    # RBF + 白噪声核 (经典稀疏组合)
    C(0.1, (1e-3, 1e1)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e1)) +
    WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e1)),

    # Matern + 白噪声核
    C(0.1, (1e-3, 1e1)) * Matern(length_scale=0.5, length_scale_bounds=(1e-2, 1e1), nu=1.5) +
    WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e1)),

    # 非常稀疏的配置 - 小方差和固定参数
    C(0.01, (1e-4, 1e0)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10)),

    # 针对小数据集的稀疏配置
    C(0.05, (1e-3, 1e0)) * Matern(length_scale=0.3, length_scale_bounds=(1e-2, 5), nu=0.5)
]

# 核函数描述
kernel_descriptions = [
    "Sparse RBF Kernel (fixed scale)",
    "Sparse Matern Kernel (nu=1.5, fixed scale)",
    "Sparse Rational Quadratic (alpha=10)",
    "Sparse RBF + White Noise",
    "Sparse Matern + White Noise",
    "Very Sparse RBF (small var)",
    "Sparse Matern (nu=0.5, small scale)"
]

# 稀疏高斯过程的参数网格
param_grid = {
    'gaussian__kernel': kernels,
    'gaussian__alpha': [1e-5, 1e-3],  # 使用较小的alpha值增强稀疏性
    'gaussian__n_restarts_optimizer': [5, 10],  # 增加重启次数以更好地优化稀疏模型
    'gaussian__normalize_y': [True]  # 固定y值归一化以增强稳定性
}

print(
    f"Total parameter combinations: {len(param_grid['gaussian__kernel']) * len(param_grid['gaussian__alpha']) * len(param_grid['gaussian__n_restarts_optimizer']) * len(param_grid['gaussian__normalize_y'])}")
print("Sparse Kernels being tested:")
for i, desc in enumerate(kernel_descriptions):
    print(f"{i + 1}. {desc}")

# 4. 执行网格搜索
print("\nPerforming Grid Search for Sparse Gaussian Process...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# 执行网格搜索
grid_search.fit(X_train, Y_train)

print("\nGrid Search completed!")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score (Negative MSE): {grid_search.best_score_:.6f}")

# 5. 使用最佳模型进行预测
best_model = grid_search.best_estimator_

# 预测（使用管道直接预测）
Y_pred = best_model.predict(X_test)

# 6. 为了获取不确定性估计，我们需要重新训练GaussianProcessRegressor模型
# 从最佳参数中提取GaussianProcessRegressor的参数（去掉前缀）
best_gaussian_params = {}
for key, value in grid_search.best_params_.items():
    if key.startswith('gaussian__'):
        param_name = key.replace('gaussian__', '')
        best_gaussian_params[param_name] = value

print(f"Best Sparse GaussianProcessRegressor parameters: {best_gaussian_params}")

# 使用最佳参数重新训练GaussianProcessRegressor来获取std
scaler = best_model.named_steps['scaler']
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练GaussianProcessRegressor模型用于不确定性估计
gaussian_model_for_std = GaussianProcessRegressor(**best_gaussian_params)
gaussian_model_for_std.fit(X_train_scaled, Y_train)

# 获取预测和标准差
Y_pred_std, Y_std = gaussian_model_for_std.predict(X_test_scaled, return_std=True)

# 确保预测值一致
assert np.allclose(Y_pred, Y_pred_std), "Predictions from pipeline and GaussianProcessRegressor should match"

# 7. 模型评估
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# 获取最佳核函数类型
best_kernel_index = kernels.index(best_gaussian_params['kernel']) if 'kernel' in best_gaussian_params else 0
kernel_type = kernel_descriptions[best_kernel_index]

print(f"\nBest Sparse Model Performance:")
print(f"Kernel Type: {kernel_type}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R-squared (R²): {r2:.4f}")

# 8. 网格搜索结果分析
print("\nGrid Search Results Analysis:")
results_df = pd.DataFrame(grid_search.cv_results_)

# 显示前5个最佳结果
top_5_indices = results_df['mean_test_score'].nlargest(5).index
top_results = results_df.loc[top_5_indices, ['params', 'mean_test_score', 'std_test_score']]

print("Top 5 parameter combinations:")
for i, (idx, row) in enumerate(top_results.iterrows()):
    print(f"{i + 1}. MSE: {-row['mean_test_score']:.6f} ± {row['std_test_score']:.6f}")
    # 简化参数显示，避免打印完整的核函数
    params = row['params']
    simplified_params = {}
    for key, value in params.items():
        if key == 'gaussian__kernel':
            # 使用核函数索引而不是完整对象
            kernel_idx = kernels.index(value)
            simplified_params[key] = f"Sparse_Kernel_{kernel_idx + 1}"
        else:
            simplified_params[key] = value
    print(f"   Params: {simplified_params}")

# 9. 可视化结果
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Sparse Gaussian Process Regression with Grid Search (Best Kernel: {kernel_type})',
             fontsize=16, fontweight='bold')

# 9.1 预测值 vs 真实值散点图
axes[0, 0].scatter(Y_test, Y_pred, alpha=0.6, color='steelblue', s=50)
axes[0, 0].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()],
                'r--', lw=2, label='Perfect prediction')
axes[0, 0].set_xlabel('True Values')
axes[0, 0].set_ylabel('Predicted Values')
axes[0, 0].set_title('Predicted vs True Values')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}\nKernel = {kernel_type}',
                transform=axes[0, 0].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')

# 9.2 残差图
residuals = Y_test - Y_pred
axes[0, 1].scatter(Y_pred, residuals, alpha=0.6, color='coral', s=50)
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Values')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Analysis')
axes[0, 1].grid(True, alpha=0.3)

# 9.3 不确定性可视化
sample_indices = range(min(20, len(Y_test)))
axes[0, 2].errorbar(sample_indices, Y_pred[:20], yerr=Y_std[:20],
                    fmt='o', alpha=0.7, color='green', ecolor='lightcoral',
                    elinewidth=2, capsize=4, label='Prediction ± Std', markersize=6)
axes[0, 2].plot(sample_indices, Y_test[:20], 's', alpha=0.8, color='red',
                markersize=5, label='True values')
axes[0, 2].set_xlabel('Sample Index')
axes[0, 2].set_ylabel('Values')
axes[0, 2].set_title('Prediction Uncertainty')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 9.4 不同核函数的性能比较
kernel_performance = {}

# 使用索引而不是字符串比较
for i, kernel in enumerate(kernels):
    # 找到使用这个核函数的所有参数组合
    kernel_scores = []
    for idx, row in results_df.iterrows():
        params = row['params']
        if params['gaussian__kernel'] == kernel:
            kernel_scores.append(-row['mean_test_score'])

    if kernel_scores:
        kernel_performance[kernel_descriptions[i]] = min(kernel_scores)

# 按性能排序
if kernel_performance:
    kernel_performance = dict(sorted(kernel_performance.items(), key=lambda x: x[1]))

    axes[1, 0].bar(range(len(kernel_performance)), list(kernel_performance.values()),
                   color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightblue', 'pink', 'lightyellow'],
                   alpha=0.7)
    axes[1, 0].set_xticks(range(len(kernel_performance)))
    axes[1, 0].set_xticklabels(list(kernel_performance.keys()), rotation=45, ha='right')
    axes[1, 0].set_xlabel('Kernel Type')
    axes[1, 0].set_ylabel('Best MSE (lower is better)')
    axes[1, 0].set_title('Best Performance by Sparse Kernel Type')
    axes[1, 0].grid(True, alpha=0.3)

    # 标记最佳核函数
    if kernel_type in kernel_performance:
        best_index = list(kernel_performance.keys()).index(kernel_type)
        axes[1, 0].axvline(x=best_index, color='red', linestyle='--', alpha=0.8, linewidth=2)
        axes[1, 0].text(best_index, max(kernel_performance.values()) * 0.9, 'Selected',
                        ha='center', va='center', fontweight='bold', color='red')
else:
    axes[1, 0].text(0.5, 0.5, 'No kernel performance data available',
                    ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Sparse Kernel Performance Comparison')

# 9.5 预测误差分布
axes[1, 1].hist(residuals, bins=15, alpha=0.7, color='purple',
                edgecolor='black', density=True)
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Prediction Error')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Prediction Error Distribution')
axes[1, 1].grid(True, alpha=0.3)

# 添加正态分布曲线
x = np.linspace(residuals.min(), residuals.max(), 100)
pdf = stats.norm.pdf(x, residuals.mean(), residuals.std())
axes[1, 1].plot(x, pdf, 'k-', linewidth=2, label='Normal fit')
axes[1, 1].legend()

# 9.6 最佳参数展示
best_params_clean = {
    'Kernel Type': kernel_type,
    'Alpha': best_gaussian_params.get('alpha', 'N/A'),
    'Restarts': best_gaussian_params.get('n_restarts_optimizer', 'N/A'),
    'Normalize Y': best_gaussian_params.get('normalize_y', 'N/A')
}

axes[1, 2].axis('off')
table_data = [[k, str(v)] for k, v in best_params_clean.items()]
table = axes[1, 2].table(cellText=table_data,
                         colLabels=['Parameter', 'Best Value'],
                         cellLoc='center', loc='center',
                         bbox=[0.2, 0.2, 0.6, 0.6])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
axes[1, 2].set_title('Best Sparse Hyperparameters')

plt.tight_layout()
plt.show()

# 10. 输出详细结果
print(f"\n{'=' * 60}")
print("DETAILED RESULTS SUMMARY")
print(f"{'=' * 60}")

print(f"\nBest Sparse Model Configuration:")
for param, value in best_params_clean.items():
    print(f"{param}: {value}")

print(f"\nModel Performance on Test Set:")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R²: {r2:.4f}")

# 11. 置信区间分析
confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)
ci_lower = Y_pred - z_score * Y_std
ci_upper = Y_pred + z_score * Y_std
coverage = np.mean((Y_test >= ci_lower) & (Y_test <= ci_upper))

print(f"\nUncertainty Analysis:")
print(f"95% Confidence Interval Coverage: {coverage * 100:.2f}%")
print(f"Mean prediction std: {Y_std.mean():.6f}")

# 12. 不同核函数的详细比较
print(f"\nComparison of Different Sparse Kernel Types:")
for i, desc in enumerate(kernel_descriptions):
    kernel_scores = []
    for idx, row in results_df.iterrows():
        params = row['params']
        if params['gaussian__kernel'] == kernels[i]:
            kernel_scores.append(-row['mean_test_score'])

    if kernel_scores:
        best_mse = min(kernel_scores)
        avg_mse = np.mean(kernel_scores)
        print(f"{desc}: Best MSE = {best_mse:.6f}, Average MSE = {avg_mse:.6f}")

print(f"\nSparse Gaussian Process Grid Search completed successfully!")
print(f"Total parameter combinations tested: {len(results_df)}")

# 13. 稀疏性分析
print(f"\nSparsity Analysis:")
print("Sparse Gaussian Process characteristics:")
print("- Uses fixed or constrained length scales to induce sparsity")
print("- Smaller variance parameters to reduce model complexity")
print("- White kernel components to handle noise separately")
print("- Optimized for small datasets with limited samples")