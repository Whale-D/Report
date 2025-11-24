import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import BayesianRidge
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

from 葫芦娃救爷爷.data_module import load_boston_data,load_friedman_data
X_train, X_test, Y_train, Y_test = load_friedman_data(
        n=50
    )

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")


# data_path = 'filtered_sampled_data.csv'
# output_columns = ['Stage2.Output.Measurement14.U.Actual']
# n_samples = 150
# drop_columns = []  # 可添加需要删除的列
#
# X_train, X_test, Y_train, Y_test = prepare_data(
#     data_path=data_path,
#     output_columns=output_columns,
#     n_samples=n_samples,
#     drop_columns=drop_columns
# )

# 3. 定义管道和参数网格
print("\nSetting up Grid Search with Polynomial Features...")

# 创建管道
pipeline = Pipeline([
    ('poly', PolynomialFeatures()),  # 多项式特征
    ('scaler', StandardScaler()),    # 标准化
    ('bayesian', BayesianRidge())     # 贝叶斯回归
])

# 定义参数网格
param_grid = {
    'poly__degree': [1, 2, 3],  # 多项式次数：线性、二次、三次
    'bayesian__alpha_1': [1e-7, 1e-6, 1e-5],
    'bayesian__alpha_2': [1e-7, 1e-6, 1e-5],
    'bayesian__lambda_1': [1e-7, 1e-6, 1e-5],
    'bayesian__lambda_2': [1e-7, 1e-6, 1e-5],
    'bayesian__max_iter': [100, 300, 500]
}

print(f"Total parameter combinations: {len(param_grid['poly__degree']) * len(param_grid['bayesian__alpha_1']) * len(param_grid['bayesian__alpha_2']) * len(param_grid['bayesian__lambda_1']) * len(param_grid['bayesian__lambda_2']) * len(param_grid['bayesian__max_iter'])}")

# 4. 执行网格搜索
print("\nPerforming Grid Search...")
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

# 获取多项式转换后的特征数量
poly_transformer = best_model.named_steps['poly']
X_poly_train = poly_transformer.transform(X_train)
print(f"\nPolynomial features (degree {poly_transformer.degree}): {X_poly_train.shape[1]} features")

# 预测（使用管道直接预测）
Y_pred = best_model.predict(X_test)

# 6. 为了获取不确定性估计，我们需要重新训练BayesianRidge模型
# 从最佳参数中提取BayesianRidge的参数（去掉前缀）
best_bayesian_params = {}
for key, value in grid_search.best_params_.items():
    if key.startswith('bayesian__'):
        param_name = key.replace('bayesian__', '')
        best_bayesian_params[param_name] = value

print(f"Best BayesianRidge parameters: {best_bayesian_params}")

# 使用最佳参数重新训练BayesianRidge来获取std
scaler = best_model.named_steps['scaler']
X_poly_test = poly_transformer.transform(X_test)
X_test_scaled = scaler.transform(X_poly_test)

# 训练BayesianRidge模型用于不确定性估计
bayesian_model_for_std = BayesianRidge(**best_bayesian_params)
X_poly_train_scaled = scaler.transform(poly_transformer.transform(X_train))
bayesian_model_for_std.fit(X_poly_train_scaled, Y_train)

# 获取预测和标准差
Y_pred_std, Y_std = bayesian_model_for_std.predict(X_test_scaled, return_std=True)

# 确保预测值一致
assert np.allclose(Y_pred, Y_pred_std), "Predictions from pipeline and BayesianRidge should match"

# 7. 模型评估
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"\nBest Model Performance:")
print(f"Polynomial Degree: {poly_transformer.degree}")
print(f"Root Mean Squared Error (RMSE): {mae:.6f}")
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
    print(f"{i+1}. MSE: {-row['mean_test_score']:.6f} ± {row['std_test_score']:.6f}")
    print(f"   Params: {row['params']}")

# 9. 可视化结果
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Bayesian Regression with Grid Search (Degree {poly_transformer.degree})',
             fontsize=16, fontweight='bold')

# 9.1 预测值 vs 真实值散点图
axes[0, 0].scatter(Y_test, Y_pred, alpha=0.6, color='steelblue', s=50)
axes[0, 0].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()],
                'r--', lw=2, label='Perfect prediction')
axes[0, 0].set_xlabel('True House Price ($K)')
axes[0, 0].set_ylabel('Predicted Price ($K)')
axes[0, 0].set_title('Predicted vs True Values')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}\nDegree = {poly_transformer.degree}', 
                transform=axes[0, 0].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')

# 9.2 残差图
residuals = Y_test - Y_pred
axes[0, 1].scatter(Y_pred, residuals, alpha=0.6, color='coral', s=50)
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Price ($K)')
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
axes[0, 2].set_ylabel('House Price ($K)')
axes[0, 2].set_title('Prediction Uncertainty')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 9.4 不同多项式次数的性能比较
degree_performance = {}
for degree in [1, 2, 3]:
    degree_mask = results_df['param_poly__degree'] == degree
    if degree_mask.any():
        best_score_for_degree = -results_df[degree_mask]['mean_test_score'].max()
        degree_performance[degree] = best_score_for_degree

axes[1, 0].bar(degree_performance.keys(), degree_performance.values(), 
               color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
axes[1, 0].set_xlabel('Polynomial Degree')
axes[1, 0].set_ylabel('Best MSE (lower is better)')
axes[1, 0].set_title('Best Performance by Polynomial Degree')
axes[1, 0].grid(True, alpha=0.3)

# 标记最佳degree
best_degree = poly_transformer.degree
axes[1, 0].axvline(x=best_degree, color='red', linestyle='--', alpha=0.8, linewidth=2)
axes[1, 0].text(best_degree, max(degree_performance.values()) * 0.9, 'Selected', 
                ha='center', va='center', fontweight='bold', color='red')

# 9.5 预测误差分布
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

# 9.6 最佳参数展示
best_params_clean = {
    'Degree': poly_transformer.degree,
    'Alpha1': best_bayesian_params.get('alpha_1', 'N/A'),
    'Alpha2': best_bayesian_params.get('alpha_2', 'N/A'),
    'Lambda1': best_bayesian_params.get('lambda_1', 'N/A'),
    'Lambda2': best_bayesian_params.get('lambda_2', 'N/A'),
    'Iterations': best_bayesian_params.get('max_iter', 'N/A')
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
axes[1, 2].set_title('Best Hyperparameters')

plt.tight_layout()
plt.show()

# 10. 输出详细结果
print(f"\n{'='*60}")
print("DETAILED RESULTS SUMMARY")
print(f"{'='*60}")

print(f"\nBest Model Configuration:")
for param, value in best_params_clean.items():
    print(f"{param}: {value}")

print(f"\nModel Performance on Test Set:")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.4f}")

# 11. 置信区间分析
confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)
ci_lower = Y_pred - z_score * Y_std
ci_upper = Y_pred + z_score * Y_std
coverage = np.mean((Y_test >= ci_lower) & (Y_test <= ci_upper))

print(f"\nUncertainty Analysis:")
print(f"95% Confidence Interval Coverage: {coverage*100:.2f}%")
print(f"Mean prediction std: ${Y_std.mean():.4f}K")

# 12. 不同degree的详细比较
print(f"\nComparison of Different Polynomial Degrees:")
for degree in [1, 2, 3]:
    degree_mask = results_df['param_poly__degree'] == degree
    if degree_mask.any():
        degree_results = results_df[degree_mask]
        best_mse = -degree_results['mean_test_score'].max()
        avg_mse = -degree_results['mean_test_score'].mean()
        print(f"Degree {degree}: Best MSE = {best_mse:.6f}, Average MSE = {avg_mse:.6f}")

print(f"\nGrid Search completed successfully!")
print(f"Total parameter combinations tested: {len(results_df)}")