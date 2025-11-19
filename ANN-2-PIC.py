import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# 1. 数据准备
df = pd.read_csv('pipe_data.csv')
df = df.dropna()

X = df[['inlet_velocity', 'degree']]
y = df[['main_mass_flow', 'branch_mass_flow']]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=168)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=168)


# 定义MRE计算函数
def mean_relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


# 创建特定参数的模型
def create_lbfgs_model():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            solver='lbfgs',
            activation='tanh',
            hidden_layer_sizes=(100,),
            alpha=0.0001,
            random_state=168,
            max_iter=1000,  # 增加迭代次数确保收敛
            early_stopping=False,  # LBFGS不使用早停
            verbose=True  # 显示训练过程
        ))
    ])
    return pipeline


# 创建并训练模型
print("\n训练LBFGS模型...")
model = create_lbfgs_model()
model.fit(X_train, y_train.values)

# 评估模型
print("\n模型评估结果:")
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

mre_val = mean_relative_error(y_val.values, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)
mre_test = mean_relative_error(y_test.values, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"验证集 - MRE: {mre_val:.4f}, R²: {r2_val:.4f}")
print(f"测试集 - MRE: {mre_test:.4f}, R²: {r2_test:.4f}")

# 计算相对误差
relative_error_main = np.abs((y_test.iloc[:, 0] - y_pred_test[:, 0]) / y_test.iloc[:, 0])
relative_error_branch = np.abs((y_test.iloc[:, 1] - y_pred_test[:, 1]) / y_test.iloc[:, 1])

# 保存预测结果
results_df = pd.DataFrame({
    'inlet_velocity': X_test.iloc[:, 0],
    'degree': X_test.iloc[:, 1],
    'main_mass_flow_true': y_test.iloc[:, 0],
    'main_mass_flow_pred': y_pred_test[:, 0],
    'main_mass_flow_error': relative_error_main,
    'branch_mass_flow_true': y_test.iloc[:, 1],
    'branch_mass_flow_pred': y_pred_test[:, 1],
    'branch_mass_flow_error': relative_error_branch
})

# 创建PDF文件保存所有图表
with PdfPages('model_results.pdf') as pdf:
    # 1. 预测值与真实值对比图
    plt.figure(figsize=(12, 10))

    # 主质量流量对比
    plt.subplot(2, 2, 1)
    plt.scatter(y_test.iloc[:, 0], y_pred_test[:, 0], alpha=0.5)
    plt.plot([y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()],
             [y_test.iloc[:, 0].min(), y_test.iloc[:, 0].max()], 'r--')
    plt.xlabel('TURE')
    plt.ylabel('PRED')
    plt.title('main-flow: ture vs pred')

    # 分支质量流量对比
    plt.subplot(2, 2, 2)
    plt.scatter(y_test.iloc[:, 1], y_pred_test[:, 1], alpha=0.5)
    plt.plot([y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()],
             [y_test.iloc[:, 1].min(), y_test.iloc[:, 1].max()], 'r--')
    plt.xlabel('ture')
    plt.ylabel('pred')
    plt.title('branch: pred vs ture')

    # 主质量流量误差分布
    plt.subplot(2, 2, 3)
    plt.hist(relative_error_main, bins=30, alpha=0.7)
    plt.xlabel('MRE')
    plt.ylabel('Frequency')
    plt.title('Distribution of main mass flow error')

    # 分支质量流量误差分布
    plt.subplot(2, 2, 4)
    plt.hist(relative_error_branch, bins=30, alpha=0.7)
    plt.xlabel('MRE')
    plt.ylabel('Frequency')
    plt.title('Distribution of branch mass flow error')

    plt.tight_layout()
    pdf.savefig()  # 保存当前图表到PDF
    plt.close()

    # 2. 特征与误差关系图
    plt.figure(figsize=(12, 6))

    # 入口速度与误差关系
    plt.subplot(1, 2, 1)
    plt.scatter(X_test.iloc[:, 0], relative_error_main, alpha=0.5, label='main-flow')
    plt.scatter(X_test.iloc[:, 0], relative_error_branch, alpha=0.5, label='branch-flow')
    plt.xlabel('inlet-v')
    plt.ylabel('MRE')
    plt.title('inlet-v & MRE')
    plt.legend()

    # 角度与误差关系
    plt.subplot(1, 2, 2)
    plt.scatter(X_test.iloc[:, 1], relative_error_main, alpha=0.5, label='main-flow')
    plt.scatter(X_test.iloc[:, 1], relative_error_branch, alpha=0.5, label='branch-flow')
    plt.xlabel('DEGREE')
    plt.ylabel('MRE')
    plt.title('DEGREE & MRE')
    plt.legend()

    plt.tight_layout()
    pdf.savefig()  # 保存当前图表到PDF
    plt.close()

# 保存结果和模型
output_csv = 'lbfgs_tanh_predictions.csv'
results_df.to_csv(output_csv, index=False)
print(f"\n预测结果已保存至: {output_csv}")

output_model = 'lbfgs_tanh_model.pkl'
joblib.dump(model, output_model)
print(f"模型已保存至: {output_model}")

# 保存模型信息到文本文件
output_txt = 'lbfgs_tanh_model_info.txt'
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write("===== 模型参数 =====\n")
    f.write("优化器: lbfgs\n")
    f.write("激活函数: tanh\n")
    f.write("隐藏层结构: (100,)\n")
    f.write("L2正则化系数: 0.0001\n")
    f.write("\n===== 模型性能 =====\n")
    f.write(f"验证集 MRE: {mre_val:.4f}\n")
    f.write(f"验证集 R²: {r2_val:.4f}\n")
    f.write(f"测试集 MRE: {mre_test:.4f}\n")
    f.write(f"测试集 R²: {r2_test:.4f}\n")
    f.write("\n===== 前5个样本预测 =====\n")
    f.write(results_df.head().to_string(index=False))

print(f"模型信息已保存至: {output_txt}")
print(f"可视化结果已保存至: model_results.pdf")