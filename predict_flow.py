import joblib
import pandas as pd

# 载入模型
model = joblib.load('lbfgs_tanh_model.pkl')

# 输入
X_new = pd.DataFrame({
    'inlet_velocity': [9.9],   # 入口风速
    'degree': [30.0]           # 阀角度
})

# ③ 预测
main_flow, branch_flow = model.predict(X_new)[0]
print(f"主管质量流量: {main_flow:.3f}")
print(f"支管质量流量: {branch_flow:.3f}")
