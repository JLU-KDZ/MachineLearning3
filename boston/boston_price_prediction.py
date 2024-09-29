import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import SGDRegressor

# 加载数据
data = pd.read_csv('boston.csv')
X = data.drop('MEDV', axis=1).values
y = data['MEDV'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义计算MSE和R2的函数
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def compute_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


# 线性回归类
class MyLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        for _ in range(self.iterations):
            gradients = (2 / m) * X.T.dot(X.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        return X.dot(self.theta)


# 绘制散点图和拟合曲线的函数
def plot_actual_vs_predicted(y_actual, y_predicted, title):
    plt.figure(figsize=(8, 4))
    plt.scatter(y_actual, y_predicted, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # 绘制参考线
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()


# 绘制原始价格和预测价格的折线图函数
def plot_original_vs_predicted(y_original, y_predicted, title):
    plt.figure(figsize=(10, 5))
    plt.plot(y_original, label='Actual Price', color='blue', marker='o', linestyle='-')
    plt.plot(y_predicted, label='Predicted Price', color='red', marker='x', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# 训练和测试原始数据的线性回归模型（BGD）
lin_reg_original = LinearRegression()
lin_reg_original.fit(X_train, y_train)
y_pred_original = lin_reg_original.predict(X_test)
mse_original_bgd = compute_mse(y_test, y_pred_original)
r2_original_bgd = compute_r2(y_test, y_pred_original)

# 归一化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练和测试归一化数据的线性回归模型（BGD）
lin_reg_scaled = LinearRegression()
lin_reg_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = lin_reg_scaled.predict(X_test_scaled)
mse_scaled_bgd = compute_mse(y_test, y_pred_scaled)
r2_scaled_bgd = compute_r2(y_test, y_pred_scaled)

# 训练和测试原始数据的线性回归模型（SGD）
sgd_reg_original = SGDRegressor(max_iter=100000, tol=1e-3, eta0=0.00001, random_state=42, shuffle=True)
sgd_reg_original.fit(X_train, y_train)
y_pred_sgd_original = sgd_reg_original.predict(X_test)
mse_original_sgd = compute_mse(y_test, y_pred_sgd_original)
r2_original_sgd = compute_r2(y_test, y_pred_sgd_original)

# 训练和测试归一化数据的线性回归模型（SGD）
sgd_reg_scaled = SGDRegressor(max_iter=100000, tol=1e-3, eta0=0.00001, random_state=42, shuffle=True)
sgd_reg_scaled.fit(X_train_scaled, y_train)
y_pred_sgd_scaled = sgd_reg_scaled.predict(X_test_scaled)
mse_scaled_sgd = compute_mse(y_test, y_pred_sgd_scaled)
r2_scaled_sgd = compute_r2(y_test, y_pred_sgd_scaled)

# 训练和测试归一化数据的岭回归模型（SGD）
ridge_reg_scaled_sgd = Ridge(alpha=1.0)
ridge_reg_scaled_sgd.fit(X_train_scaled, y_train)
y_pred_ridge_sgd = ridge_reg_scaled_sgd.predict(X_test_scaled)
mse_ridge_sgd = compute_mse(y_test, y_pred_ridge_sgd)
r2_ridge_sgd = compute_r2(y_test, y_pred_ridge_sgd)

# 训练和测试归一化数据的LASSO回归模型（SGD）
lasso_reg_scaled_sgd = Lasso(alpha=0.1)
lasso_reg_scaled_sgd.fit(X_train_scaled, y_train)
y_pred_lasso_sgd = lasso_reg_scaled_sgd.predict(X_test_scaled)
mse_lasso_sgd = compute_mse(y_test, y_pred_lasso_sgd)
r2_lasso_sgd = compute_r2(y_test, y_pred_lasso_sgd)

# 输出MSE和R2
print(f'MSE for Original Data + BGD: {mse_original_bgd:.5f}')
print(f'MSE for Normalized Data + BGD: {mse_scaled_bgd:.5f}')
print(f'MSE for Original Data + SGD: {mse_original_sgd:.5f}')
print(f'MSE for Normalized Data + SGD: {mse_scaled_sgd:.5f}')
print(f'MSE for Ridge Regression + SGD: {mse_ridge_sgd:.5f}')
print(f'MSE for Lasso Regression + SGD: {mse_lasso_sgd:.5f}')

print(f'R2 for Original Data + BGD: {r2_original_bgd:.5f}')
print(f'R2 for Normalized Data + BGD: {r2_scaled_bgd:.5f}')
print(f'R2 for Original Data + SGD: {r2_original_sgd:.5f}')
print(f'R2 for Normalized Data + SGD: {r2_scaled_sgd:.5f}')
print(f'R2 for Ridge Regression + SGD: {r2_ridge_sgd:.5f}')
print(f'R2 for Lasso Regression + SGD: {r2_lasso_sgd:.5f}')

# 结果对比
models = [
    'Original + BGD',
    'Normalized + BGD',
    'Original + SGD',
    'Normalized + SGD',
    'Ridge + SGD',
    'Lasso + SGD'
]
mses = [
    mse_original_bgd,
    mse_scaled_bgd,
    mse_original_sgd,
    mse_scaled_sgd,
    mse_ridge_sgd,
    mse_lasso_sgd
]
r2s = [
    r2_original_bgd,
    r2_scaled_bgd,
    r2_original_sgd,
    r2_scaled_sgd,
    r2_ridge_sgd,
    r2_lasso_sgd
]

# 创建MSE图表
plt.figure(figsize=(10, 6))
plt.bar(models, mses, color=['#69b3b2', '#a6cee3', '#1c6399', '#6a51a3', '#e41a1c', '#377eb8'])
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('Model Comparison - MSE')
plt.xticks(rotation=45)
for i, v in enumerate(mses):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center')  # 显示MSE数值
plt.tight_layout()
plt.show()

# 创建R2图表
plt.figure(figsize=(10, 6))
plt.bar(models, r2s, color=['#b2df8a', '#a6cee3', '#1c6399', '#6a51a3', '#e41a1c', '#4575b4'])
plt.xlabel('Model')
plt.ylabel('R2 Score')
plt.title('Model Comparison - R2 Score')
plt.xticks(rotation=45)
for i, v in enumerate(r2s):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')  # 显示R2数值
plt.tight_layout()
plt.show()

# 系数对比
coefficients = {
    'Original BGD': lin_reg_original.coef_,
    'Normalized BGD': lin_reg_scaled.coef_,
    'Original SGD': sgd_reg_original.coef_,
    'Normalized SGD': sgd_reg_scaled.coef_,
    'Ridge SGD': ridge_reg_scaled_sgd.coef_,
    'Lasso SGD': lasso_reg_scaled_sgd.coef_
}
for model, coef in coefficients.items():
    print(f'{model} coefficients: {coef}')

# 绘制实际值与预测值的散点图和拟合曲线
plot_actual_vs_predicted(y_test, y_pred_original, 'Original Data + BGD')
plot_actual_vs_predicted(y_test, y_pred_scaled, 'Normalized Data + BGD')
plot_actual_vs_predicted(y_test, y_pred_sgd_original, 'Original Data + SGD')
plot_actual_vs_predicted(y_test, y_pred_sgd_scaled, 'Normalized Data + SGD')
plot_actual_vs_predicted(y_test, y_pred_ridge_sgd, 'Ridge Regression + SGD')
plot_actual_vs_predicted(y_test, y_pred_lasso_sgd, 'Lasso Regression + SGD')

# 绘制原始价格和预测价格的折线图
plot_original_vs_predicted(y_test, y_pred_original, 'Original Data + BGD')
plot_original_vs_predicted(y_test, y_pred_scaled, 'Normalized Data + BGD')
plot_original_vs_predicted(y_test, y_pred_sgd_original, 'Original Data + SGD')
plot_original_vs_predicted(y_test, y_pred_sgd_scaled, 'Normalized Data + SGD')
plot_original_vs_predicted(y_test, y_pred_ridge_sgd, 'Ridge Regression + SGD')
plot_original_vs_predicted(y_test, y_pred_lasso_sgd, 'Lasso Regression + SGD')
