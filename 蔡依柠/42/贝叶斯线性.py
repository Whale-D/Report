import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
import scipy.stats as stats

# 设置图形样式（使用英文字体避免中文问题）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

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

# 2. 数据预处理 - 只取前800个样本保持简单
X = X[:50]
Y = Y[:50]

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.3, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# 3. 贝叶斯岭回归
print("\nTraining Bayesian Regression Model...")
bayesian_model = BayesianRidge(
    max_iter=300,
    tol=1e-3,
    alpha_1=1e-6,
    alpha_2=1e-6,
    lambda_1=1e-6,
    lambda_2=1e-6
)

# 训练模型
bayesian_model.fit(X_train, Y_train)

# 预测
Y_pred, Y_std = bayesian_model.predict(X_test, return_std=True)

# 4. 模型评估
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"\nModel Performance:")
print(f"Root Mean Squared Error (RMSE): {mae:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R-squared (R²): {r2:.4f}")

# 5. 可视化结果
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Bayesian Regression - California Housing Price Prediction',
             fontsize=16, fontweight='bold')

# 5.1 预测值 vs 真实值散点图
axes[0, 0].scatter(Y_test, Y_pred, alpha=0.6, color='steelblue', s=50)
axes[0, 0].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()],
                'r--', lw=2, label='Perfect prediction')
axes[0, 0].set_xlabel('True House Price ($K)')
axes[0, 0].set_ylabel('Predicted Price ($K)')
axes[0, 0].set_title('Predicted vs True Values')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 添加R²到图中
axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top')

# 5.2 残差图
residuals = Y_test - Y_pred
axes[0, 1].scatter(Y_pred, residuals, alpha=0.6, color='coral', s=50)
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Price ($K)')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Analysis')
axes[0, 1].grid(True, alpha=0.3)

# 5.3 不确定性可视化（前30个测试样本）
sample_indices = range(min(30, len(Y_test)))
axes[0, 2].errorbar(sample_indices, Y_pred[:30], yerr=Y_std[:30],
                   fmt='o', alpha=0.7, color='green', ecolor='lightcoral',
                   elinewidth=2, capsize=4, label='Prediction ± Std', markersize=6)
axes[0, 2].plot(sample_indices, Y_test[:30], 's', alpha=0.8, color='red',
               markersize=5, label='True values')
axes[0, 2].set_xlabel('Sample Index')
axes[0, 2].set_ylabel('House Price ($K)')
axes[0, 2].set_title('Prediction Uncertainty (First 30 Samples)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 5.4 特征重要性（系数绝对值）
feature_importance = np.abs(bayesian_model.coef_)
sorted_idx = np.argsort(feature_importance)[::-1]

# 简化特征名称显示
short_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
               'Population', 'AveOccup', 'Latitude', 'Longitude']

axes[1, 0].barh(range(len(feature_importance)), feature_importance[sorted_idx],
               color='lightseagreen', alpha=0.7)
axes[1, 0].set_yticks(range(len(feature_importance)))
axes[1, 0].set_yticklabels([short_names[i] for i in sorted_idx])
axes[1, 0].set_xlabel('Coefficient Absolute Value')
axes[1, 0].set_title('Feature Importance')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 5.5 预测误差分布
axes[1, 1].hist(residuals, bins=15, alpha=0.7, color='purple',
                edgecolor='black', density=True)
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Prediction Error ($K)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Prediction Error Distribution')
axes[1, 1].grid(True, alpha=0.3)

# 添加正态分布曲线
x = np.linspace(residuals.min(), residuals.max(), 100)
pdf = stats.norm.pdf(x, residuals.mean(), residuals.std())
axes[1, 1].plot(x, pdf, 'k-', linewidth=2, label='Normal fit')
axes[1, 1].legend()

# 5.6 模型系数
x_pos = np.arange(len(bayesian_model.coef_))
bars = axes[1, 2].bar(x_pos, bayesian_model.coef_, color='orange', alpha=0.7)
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels([f'F{i+1}' for i in range(len(short_names))])
axes[1, 2].set_xlabel('Feature')
axes[1, 2].set_ylabel('Coefficient Value')
axes[1, 2].set_title('Model Coefficients')
axes[1, 2].grid(True, alpha=0.3)

# 为正值和负值系数着色
for i, bar in enumerate(bars):
    if bayesian_model.coef_[i] < 0:
        bar.set_color('red')
    else:
        bar.set_color('blue')

plt.tight_layout()
plt.show()

# 6. 输出模型详细信息
print(f"\nBayesian Regression Model Parameters:")
print(f"Coefficients (weights): {bayesian_model.coef_}")
print(f"Intercept: {bayesian_model.intercept_:.4f}")
print(f"Alpha (precision): {bayesian_model.alpha_:.6f}")
print(f"Lambda (regularization): {bayesian_model.lambda_:.6f}")

# 7. 特征重要性排序
print(f"\nFeature Importance Ranking:")
for i, idx in enumerate(sorted_idx):
    print(f"{i+1:2d}. {short_names[idx]:10} : {feature_importance[idx]:.4f}")

# 8. 预测不确定性统计
print(f"\nPrediction Uncertainty Statistics:")
print(f"Mean prediction std: ${Y_std.mean():.4f}K")
print(f"Std of prediction std: ${Y_std.std():.4f}K")
print(f"Prediction std range: [${Y_std.min():.4f}K, ${Y_std.max():.4f}K]")

# 9. 置信区间分析
confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)
ci_lower = Y_pred - z_score * Y_std
ci_upper = Y_pred + z_score * Y_std
coverage = np.mean((Y_test >= ci_lower) & (Y_test <= ci_upper))

print(f"\n{confidence_level*100:.0f}% Confidence Interval Coverage: {coverage*100:.2f}%")

# 10. 额外：前10个样本的详细预测
print(f"\nDetailed predictions for first 10 test samples:")
print("Index | True Value | Prediction | Std Dev | 95% CI Lower | 95% CI Upper | In CI")
print("-" * 80)
for i in range(min(10, len(Y_test))):
    in_ci = "Yes" if (Y_test[i] >= ci_lower[i]) and (Y_test[i] <= ci_upper[i]) else "No"
    print(f"{i:5d} | ${Y_test[i]:8.3f}K | ${Y_pred[i]:8.3f}K | ${Y_std[i]:6.3f}K | "
          f"${ci_lower[i]:8.3f}K | ${ci_upper[i]:8.3f}K | {in_ci:>5}")