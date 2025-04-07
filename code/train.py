import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 数据集类，用于加载和预处理RCS数据
class RCSDataset(Dataset):
    def __init__(self, data_dir='DATA_01', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.label_map = {}
        
        # 加载所有数据
        self._load_data()
        
    def _load_data(self):
        #加载DATA_01文件夹下的所有mat文件数据
        label_idx = 0
        
        # 遍历所有目标文件夹（1-10）
        for target_folder in sorted(os.listdir(self.data_dir)):
            target_path = os.path.join(self.data_dir, target_folder)
            
            if os.path.isdir(target_path):
                # 将文件夹名映射为数字标签
                self.label_map[target_folder] = label_idx
                
                # 遍历该目标文件夹下的所有frame数据
                for frame_file in os.listdir(target_path):
                    if frame_file.endswith('.mat'):
                        file_path = os.path.join(target_path, frame_file)
                        self.samples.append(file_path)
                        self.labels.append(label_idx)
                        
                label_idx += 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本数据"""
        file_path = self.samples[idx]
        label = self.labels[idx]
        
        # 加载mat文件
        mat_data = sio.loadmat(file_path)
        
        # 提取mat文件中的变量
        frame_eh = mat_data['frame_Eh']  # 极化1的RCS数据
        frame_ev = mat_data['frame_Ev']  # 极化2的RCS数据
        
        # 获取数据的维度
        eh_shape = frame_eh.shape
        ev_shape = frame_ev.shape
        
        # 简化处理：只取第一列数据（对应一个方位角和俯仰角）
        # 这样可以将维度降到可接受的范围
        if len(eh_shape) > 1 and eh_shape[1] > 0:
            frame_eh = frame_eh[:, 0]
        
        if len(ev_shape) > 1 and ev_shape[1] > 0:
            frame_ev = frame_ev[:, 0]
        
        # 处理复数数据，分离实部和虚部
        eh_real = np.real(frame_eh)
        eh_imag = np.imag(frame_eh)
        ev_real = np.real(frame_ev)
        ev_imag = np.imag(frame_ev)
        
        # 将所有特征组合成一个特征向量
        # 只使用两种极化的RCS数据的实部和虚部
        features = np.concatenate([
            eh_real.flatten(), eh_imag.flatten(),
            ev_real.flatten(), ev_imag.flatten()
        ])
        
        # 转换为PyTorch张量
        features = torch.FloatTensor(features)
        
        # 创建one-hot标签
        label_onehot = torch.zeros(10)
        label_onehot[label] = 1.0
        
        if self.transform:
            features = self.transform(features)
            
        return features, label_onehot

# 数据预处理和增强类
class RCSTransform:
    def __init__(self, normalize=True, augment=False, noise_level=0.01):
        self.normalize = normalize
        self.augment = augment
        self.noise_level = noise_level
        
    def __call__(self, x):
        # 标准化数据
        if self.normalize:
            mean = x.mean()
            std = x.std()
            if std > 0:
                x = (x - mean) / std
                
        # 数据增强：添加随机噪声
        if self.augment:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
            
        return x

# 数据可视化函数
def visualize_rcs_data(file_path, save_path=None):
    """可视化单个RCS数据样本"""
    mat_data = sio.loadmat(file_path)
    
    freq_hz = mat_data['FreqHz'].flatten()
    frame_eh = mat_data['frame_Eh']
    frame_ev = mat_data['frame_Ev']
    
    # 只查看第一列数据
    if frame_eh.ndim > 1 and frame_eh.shape[1] > 0:
        frame_eh_col = frame_eh[:, 0]
    else:
        frame_eh_col = frame_eh
        
    if frame_ev.ndim > 1 and frame_ev.shape[1] > 0:
        frame_ev_col = frame_ev[:, 0]
    else:
        frame_ev_col = frame_ev
    
    # 绘制幅度谱
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(freq_hz, np.abs(frame_eh_col))
    plt.title('frame_Eh Magnitude (First Column)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    
    plt.subplot(2, 2, 2)
    plt.plot(freq_hz, np.abs(frame_ev_col))
    plt.title('frame_Ev Magnitude (First Column)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    
    plt.subplot(2, 2, 3)
    plt.plot(freq_hz, np.angle(frame_eh_col))
    plt.title('frame_Eh Phase (First Column)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (rad)')
    
    plt.subplot(2, 2, 4)
    plt.plot(freq_hz, np.angle(frame_ev_col))
    plt.title('frame_Ev Phase (First Column)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (rad)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# 修改模型以适应RCS数据的特性
class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim)
        )
        self.attention = AttentionBlock(input_dim)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = self.attention(out)
        out += residual
        return F.relu(out)

class RCSClassifier(nn.Module):
    def __init__(self, input_size=802, hidden_sizes=[512, 256, 128, 64], num_classes=10):
        
        super(RCSClassifier, self).__init__()
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_sizes[0], hidden_sizes[0] // 2),
            ResidualBlock(hidden_sizes[0], hidden_sizes[0] // 2)
        ])
        
        # 中间层
        self.middle_layers = nn.ModuleList()
        for i in range(len(hidden_sizes)-1):
            self.middle_layers.append(nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.BatchNorm1d(hidden_sizes[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
    def forward(self, x):
        x = self.input_layer(x)
        
        # 残差连接
        for block in self.residual_blocks:
            x = block(x)
        
        # 中间层
        for layer in self.middle_layers:
            x = layer(x)
        
        return self.output_layer(x)

class Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # 使用AdamW优化器，添加权重衰减
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # 使用余弦退火学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=200,
            eta_min=0.00001
        )
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.argmax(dim=1)).sum().item()
            total += target.size(0)
            
        # 更新学习率
        self.scheduler.step()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target.argmax(dim=1)).sum().item()
                total += target.size(0)
                
        return total_loss / len(dataloader), 100. * correct / total
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# 主函数，用于演示如何使用这些类
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
    # 创建数据集
    transform = RCSTransform(normalize=True, augment=True)
    dataset = RCSDataset(data_dir='DATA_01', transform=transform)
    
    # 获取第一个样本来确定特征维度
    first_sample, _ = dataset[0]
    input_size = first_sample.shape[0]
    
    # 划分训练集和验证集
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.labels
    )
    
    # 创建数据加载器
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        dataset, batch_size=32, sampler=train_sampler, num_workers=2
    )
    val_loader = DataLoader(
        dataset, batch_size=32, sampler=val_sampler, num_workers=2
    )
    
    # 创建模型，使用实际检测到的特征维度
    model = RCSClassifier(input_size=input_size, num_classes=10)
    model = model.to(device)
    
    # 创建训练器
    trainer = Trainer(model, device=device)
    
    # 训练模型
    epochs = 50
    best_acc = 0
    
    for epoch in range(epochs):
        print(f"\n训练周期 {epoch+1}/{epochs}")
        
        # 训练
        train_loss, train_acc = trainer.train_epoch(train_loader)
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        
        # 验证
        val_loss, val_acc = trainer.evaluate(val_loader)
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            trainer.save_model('best_model.pth')
            print(f"模型已保存，验证准确率: {val_acc:.2f}%")
    
    print(f"训练完成，最佳验证准确率: {best_acc:.2f}%")

if __name__ == "__main__":
    main() 

