import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from 拿数据 import prepare_data

# 设置图形样式（使用默认英文字体）
plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 准备数据
data_path = '葫芦娃救爷爷/filtered_sampled_data.csv'  # 根据实际情况修改
output_columns = ['Stage2.Output.Measurement14.U.Actual']
n_samples = 150
drop_columns = []

X_train, X_test, Y_train, Y_test = prepare_data(
    data_path=data_path,
    output_columns=output_columns,
    n_samples=n_samples,
    drop_columns=drop_columns
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# PCA分析
print("\nPerforming PCA analysis...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用PCA保留95%的方差
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original number of features: {X_train.shape[1]}")
print(f"Number of features after PCA: {pca.n_components_}")
print(f"Cumulative explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# 合并所有数据进行可视化
print("\nPreparing visualization...")
X_all = np.vstack([X_train_scaled, X_test_scaled])
X_all_pca = pca.transform(X_all)

# 创建数据标识（训练集 vs 测试集）
dataset_labels = np.array(['Train'] * len(Y_train) + ['Test'] * len(Y_test))

# 创建PCA相关可视化图形
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('PCA Comprehensive Analysis', fontsize=16, fontweight='bold')

# 1. PCA方差解释率
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

axes[0, 0].bar(range(1, len(explained_variance) + 1), explained_variance,
               alpha=0.6, color='skyblue', label='Individual explained variance')
axes[0, 0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                'ro-', linewidth=2, label='Cumulative explained variance')
axes[0, 0].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
axes[0, 0].set_xlabel('Number of components')
axes[0, 0].set_ylabel('Variance ratio')
axes[0, 0].set_title('PCA Explained Variance')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 前两个主成分的散点图（按数据集区分）
colors = {'Train': 'blue', 'Test': 'red'}
if X_all_pca.shape[1] >= 2:
    for dataset in ['Train', 'Test']:
        mask = dataset_labels == dataset
        axes[0, 1].scatter(X_all_pca[mask, 0], X_all_pca[mask, 1],
                           alpha=0.6, color=colors[dataset],
                           label=dataset, s=50)

    axes[0, 1].set_xlabel('1st Principal Component')
    axes[0, 1].set_ylabel('2nd Principal Component')
    axes[0, 1].set_title('Distribution of First Two Principal Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# 3. 第一和第三主成分的散点图（如果存在）
if X_all_pca.shape[1] >= 3:
    for dataset in ['Train', 'Test']:
        mask = dataset_labels == dataset
        axes[1, 0].scatter(X_all_pca[mask, 0], X_all_pca[mask, 2],
                           alpha=0.6, color=colors[dataset],
                           label=dataset, s=50)

    axes[1, 0].set_xlabel('1st Principal Component')
    axes[1, 0].set_ylabel('3rd Principal Component')
    axes[1, 0].set_title('Distribution of 1st and 3rd Principal Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

# 4. PCA分析摘要
axes[1, 1].axis('off')
n_display = min(5, len(explained_variance))
summary_text = f"""
PCA Analysis Summary:
- Original features: {X_train.shape[1]}
- Principal components: {pca.n_components_}
- Cumulative variance: {cumulative_variance[-1]:.3f}

Component Importance (top {n_display}):
{pd.DataFrame({
    'Component': range(1, n_display + 1),
    'Explained Variance': explained_variance[:n_display],
    'Cumulative Variance': cumulative_variance[:n_display]
}).to_string(index=False)}
"""

axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.show()

# 主成分的重要性分析
print("\nPrincipal Component Importance Analysis:")
component_importance = pd.DataFrame({
    'Component': range(1, len(explained_variance) + 1),
    'Explained Variance': explained_variance,
    'Cumulative Variance': cumulative_variance
})
print(component_importance.head(10))

print("\nPCA analysis completed!")