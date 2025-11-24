import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from 葫芦娃救爷爷.data_module import load_boston_data
X_train, X_test, Y_train, Y_test = load_boston_data(
        n=50
    )
# ======= 1. 构造完整数据（你已经分好的） =======
Y_all = np.concatenate([Y_train, Y_test])

# ======= 2. 绘制分布图 =======
plt.figure(figsize=(12, 6))
sns.histplot(Y_all, color='blue', kde=True, alpha=0.3, label='All Data')
sns.histplot(Y_train, color='green', kde=True, alpha=0.4, label='Train Set')
sns.histplot(Y_test, color='red', kde=True, alpha=0.4, label='Test Set')

plt.title("Distribution of Y (All Data vs Train vs Test)", fontsize=16)
plt.xlabel("Target Value (Y)")
plt.ylabel("Count / Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
