import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据预处理
# ==========================================
def load_and_preprocess_data(csv_file, window_size=5):
    df = pd.read_csv(csv_file)
    
    # 将时间戳设置为索引
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 处理缺失值 (用前一个值填充，如果没有则补0)
    df = df.ffill().fillna(0)
    
    # 提取特征
    features = df.columns.tolist()
    data = df.values
    
    # 归一化到 [0, 1] 区间 (对应论文中的 normalize 操作)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 构建时间滑动窗口 (Samples, Features, Window_Size)
    # PyTorch 的 Conv1D 期望的输入形状是 (Batch, Channels, Length)
    X = []
    for i in range(len(data_scaled) - window_size + 1):
        window = data_scaled[i:i + window_size]
        X.append(window.T) # 转置为 (Features, Window_Size)
        
    return torch.tensor(np.array(X), dtype=torch.float32), df, scaler, features

# ==========================================
# 2. 轻量级卷积自编码器 (LightCAE 模型适配)
# ==========================================
class LightCAE1D(nn.Module):
    def __init__(self, num_features):
        super(LightCAE1D, self).__init__()
        # Encoder (对应论文中的 E(X))
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder (对应论文中的 D(H))
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=num_features, kernel_size=3, padding=1),
            nn.Sigmoid() # 保证输出在 [0, 1] 之间
        )

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out

# ==========================================
# 3. 鲁棒交替优化训练 (论文 Section 3.2.2 & 5)
# ==========================================
def train_rtcae(M_tensor, num_epochs=50, outer_iters=3, epsilon_ratio=0.05):
    """
    M_tensor: 原始测量数据 (观测值)
    epsilon_ratio: 预估的异常点比例
    """
    model = LightCAE1D(num_features=M_tensor.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    # 初始化 S (稀疏异常矩阵) 为 0
    S = torch.zeros_like(M_tensor)
    
    num_elements = M_tensor.numel()
    k_threshold = int(num_elements * epsilon_ratio) # 计算非零异常值的数量阈值

    print("开始交替优化训练 (Alternating Optimization)...")
    for out_iter in range(outer_iters):
        # Step 1: 求解正常数据 X = M - S
        X = M_tensor - S
        
        # Step 2: 训练 Autoencoder (Eq. 4: 数据重构子问题)
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            C_hat = model(X) # C = D(E(X))
            loss = criterion(C_hat, X)
            loss.backward()
            optimizer.step()
            
        # 获取当前重构的正常数据 C
        model.eval()
        with torch.no_grad():
            C = model(X)
            
        # Step 3: 异常数据估计子问题 (Eq. 5 & Eq. 16)
        E = M_tensor - C  # 计算残差 E
        E_squared = E ** 2
        
        # Hard Thresholding: 提取误差最大的 epsilon 个点作为异常 S，其余置 0
        threshold_val = torch.kthvalue(E_squared.flatten(), num_elements - k_threshold).values
        
        # 更新 S (Eq. 16)
        S = torch.where(E_squared > threshold_val, E, torch.zeros_like(E))
        
        loss_val = criterion(C + S, M_tensor).item()
        print(f"Outer Iteration {out_iter+1}/{outer_iters} | 整体拟合误差 ||M - C - S||_F^2: {loss_val:.6f}")
        
    return model, C, S

# ==========================================
# 4. 执行与可视化
# ==========================================
def main():
    # 1. 加载数据
    csv_file = 'dataset.csv' # 确保你的CSV文件在同级目录
    window_size = 5
    M_tensor, df, scaler, features = load_and_preprocess_data(csv_file, window_size)
    
    # 2. 训练模型 (这里 epsilon_ratio 设为 5% 视为异常)
    model, C_tensor, S_tensor = train_rtcae(M_tensor, num_epochs=60, outer_iters=4, epsilon_ratio=0.05)
    
    # 3. 计算异常得分
    # 异常得分可以通过 S 矩阵的幅度或者残差 E 矩阵来表示
    E_tensor = M_tensor - C_tensor
    # 对特征和窗口维度求均方误差，得到时间步维度的异常得分
    anomaly_scores = torch.mean(E_tensor ** 2, dim=(1, 2)).numpy()
    
    # 补齐滑动窗口造成的时间差 (前面补0)
    scores_padded = np.pad(anomaly_scores, (window_size - 1, 0), 'constant', constant_values=0)
    df['Anomaly_Score'] = scores_padded
    
    # 动态设定阈值 (例如 3-sigma 或 百分位数)
    threshold = np.percentile(scores_padded[scores_padded > 0], 95) 
    df['Is_Anomaly'] = df['Anomaly_Score'] > threshold

    # 4. 绘图展示
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 图 1: 展示几个关键的原生指标 (比如 cpu_total 和 load_1m)
    ax1.plot(df.index, df['cpu_total'], label='CPU Total', color='blue')
    ax1.plot(df.index, df['load_1m'], label='Load 1m', color='orange')
    ax1.set_ylabel('Original Metrics')
    ax1.set_title('Microservice System Metrics (M)')
    ax1.legend(loc='upper left')
    
    # 将检测出的异常点标红
    anomalies = df[df['Is_Anomaly']]
    ax1.scatter(anomalies.index, anomalies['cpu_total'], color='red', s=50, label='Detected Anomaly', zorder=5)
    
    # 图 2: 论文中的正常基线 C (这里还原第一维特征 cpu_system 作为展示)
    # 提取重构数据 C 的最后一个窗口时间点，并反归一化
    C_numpy = C_tensor[:, :, -1].numpy() 
    C_padded = np.pad(C_numpy, ((window_size - 1, 0), (0, 0)), 'edge')
    C_inverse = scaler.inverse_transform(C_padded)
    
    ax2.plot(df.index, df['cpu_total'], label='Original CPU Total (M)', color='lightgray', linestyle='--')
    ax2.plot(df.index, C_inverse[:, features.index('cpu_total')], label='Reconstructed Normal (C)', color='green')
    ax2.set_ylabel('Reconstructed Metric')
    ax2.set_title('Extracted Normal Data Subspace (C)')
    ax2.legend(loc='upper left')

    # 图 3: 异常得分与阈值
    ax3.plot(df.index, df['Anomaly_Score'], label='Anomaly Score (||M-C||^2)', color='purple')
    ax3.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    ax3.set_ylabel('Anomaly Score')
    ax3.set_xlabel('Timestamp')
    ax3.set_title('Anomaly Score & Detection Result')
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()