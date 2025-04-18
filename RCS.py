import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import random
from torchvision import transforms
from PIL import Image, ImageOps
import scipy.stats

# 将复数数据转换为图像的函数
def complex_to_image(complex_data, target_size=(224, 224)):
    """将复数数据转换为图像表示形式"""
    magnitude = np.abs(complex_data)
    phase = np.angle(complex_data)
    
    # 归一化到0-255
    magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude) + 1e-10) * 255
    phase = (phase + np.pi) / (2 * np.pi) * 255
    
    # 转换为uint8格式
    magnitude = magnitude.astype(np.uint8)
    phase = phase.astype(np.uint8)
    
    # 创建PIL图像
    magnitude_img = Image.fromarray(magnitude)
    phase_img = Image.fromarray(phase)
    
    # 调整大小
    if target_size:
        magnitude_img = magnitude_img.resize(target_size, Image.BILINEAR)
        phase_img = phase_img.resize(target_size, Image.BILINEAR)
    
    return magnitude_img, phase_img

# 融合ISAR和PFA图像为RGB图像
def fusion_to_rgb(frame_eh, frame_ev, target_size=(224, 224)):
    """将两种极化数据融合为RGB图像"""
    # 获取幅度和相位图像
    mag_eh, phase_eh = complex_to_image(frame_eh, target_size)
    mag_ev, phase_ev = complex_to_image(frame_ev, target_size)
    
    # 创建RGB通道
    r_channel = np.array(mag_eh)
    g_channel = np.array(mag_ev)
    b_channel = np.array((np.array(phase_eh) + np.array(phase_ev)) / 2)
    
    # 合并为RGB图像
    rgb_img = np.stack((r_channel, g_channel, b_channel), axis=2)
    rgb_img = Image.fromarray(rgb_img.astype(np.uint8))
    
    return rgb_img

# 数据集类，用于加载和预处理RCS数据
class RCSDataset(Dataset):
    def __init__(self, data_dir='DATA_01', transform=None, use_fusion=True, target_size=(224, 224)):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.use_fusion = use_fusion
        self.target_size = target_size
        
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
        
        # 利用全部数据，不再只取第一列
        # 但为保证兼容性，需要处理超出处理能力的情况
        # 获取数据的维度
        eh_shape = frame_eh.shape
        ev_shape = frame_ev.shape
        
        # 多角度数据处理 - 如果是2D数据（频率 x 角度），考虑全部角度
        if len(eh_shape) > 1 and eh_shape[1] > 1:
            # 将所有角度数据合并处理或选择代表性角度
            # 简化方案：以平均值作为代表，保持维度一致性
            frame_eh_avg = np.mean(frame_eh, axis=1)
            frame_ev_avg = np.mean(frame_ev, axis=1)
            
            # 只取第一列 - 需要保持代码兼容性的情况下使用
            # frame_eh = frame_eh[:, 0]
            # frame_ev = frame_ev[:, 0]
            
            # 使用平均角度特征
            frame_eh = frame_eh_avg
            frame_ev = frame_ev_avg
        else:
            # 如果只有一个角度或一维数据，直接使用
            if len(eh_shape) > 1 and eh_shape[1] > 0:
                frame_eh = frame_eh[:, 0]
            if len(ev_shape) > 1 and ev_shape[1] > 0:
                frame_ev = frame_ev[:, 0]
        
        if self.use_fusion:
            # 将ISAR和PFA数据融合为RGB图像
            image = fusion_to_rgb(frame_eh, frame_ev, self.target_size)
            
            # 应用图像变换
            if self.transform:
                image = self.transform(image)
                
            # 创建one-hot标签
            label_onehot = torch.zeros(10)
            label_onehot[label] = 1.0
            
            return image, label_onehot
        else:
            # 处理复数数据，提取更丰富的特征
            eh_real = np.real(frame_eh)
            eh_imag = np.imag(frame_eh)
            ev_real = np.real(frame_ev)
            ev_imag = np.imag(frame_ev)
            
            # 提取更多特征以提高识别准确率
            # 计算幅度和相位
            eh_mag = np.abs(frame_eh)
            eh_phase = np.angle(frame_eh)
            ev_mag = np.abs(frame_ev)
            ev_phase = np.angle(frame_ev)
            
            # 对数变换以捕获动态范围
            eh_log_mag = np.log1p(eh_mag)
            ev_log_mag = np.log1p(ev_mag)
            
            # 将所有特征组合成一个特征向量
            features = np.concatenate([
                eh_real.flatten(), eh_imag.flatten(),
                ev_real.flatten(), ev_imag.flatten(),
                eh_mag.flatten(), eh_phase.flatten(),
                ev_mag.flatten(), ev_phase.flatten(),
                eh_log_mag.flatten(), ev_log_mag.flatten()
            ])
            
            # 转换为PyTorch张量
            features = torch.FloatTensor(features)
            
            # 创建one-hot标签
            label_onehot = torch.zeros(10)
            label_onehot[label] = 1.0
            
            if self.transform:
                features = self.transform(features)
                
            return features, label_onehot

# 图像增强转换
class ImageTransforms:
    def __init__(self, is_training=True):
        if is_training:
            self.transforms = transforms.Compose([
                transforms.RandomRotation(15),  # 随机旋转±15度
                transforms.RandomGrayscale(p=0.2),  # 20%概率转为灰度图
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.RandomVerticalFlip(),    # 随机垂直翻转
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度和对比度变化
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, x):
        return self.transforms(x)

# 数据预处理和增强类（用于原始特征向量）
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

# 角度融合模块
class AngleFusionModule(nn.Module):
    def __init__(self, input_dim):
        super(AngleFusionModule, self).__init__()
        self.conv = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(16)
        
    def forward(self, x):
        # 输入形状调整为(batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        x = F.relu(self.bn(self.conv(x)))
        return x

# 1D ResNet块
class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# BPA (Bilinear Pooling Attention)
class BPA(nn.Module):
    def __init__(self, in_channels):
        super(BPA, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels//2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 输入形状: (batch_size, channels, seq_len)
        att = self.attention(x)
        out = x * att
        return out

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x):
        # 输入形状变换: (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.multihead_attn(x, x, x)
        # 输出形状变换回: (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
        return attn_output.permute(1, 0, 2)

# 前馈神经网络层
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout以提高泛化能力
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x):
        return self.net(x)

# BiGRU模块
class BiGRUModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiGRUModule, self).__init__()
        self.bigru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        # 添加一个线性层来融合双向输出
        self.fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len, input_size)
        output, _ = self.bigru(x)
        # 应用线性层
        output = self.fc(output)
        return output

# 时序特征提取模块
class TemporalFeatureModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4):
        super(TemporalFeatureModule, self).__init__()
        
        # 第一个层归一化
        self.norm1 = nn.LayerNorm(input_dim)
        
        # 多头注意力
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads)
        
        # 第二个层归一化
        self.norm2 = nn.LayerNorm(input_dim)
        
        # 前馈层
        self.feedforward = FeedForward(input_dim, hidden_dim * 2)
        
        # 前馈层后的层归一化
        self.norm3 = nn.LayerNorm(input_dim)
        
        # BiGRU层 - 对应图中的4个BiGRU模块
        self.bigru_layers = nn.ModuleList([
            BiGRUModule(input_dim, hidden_dim//2),
            BiGRUModule(hidden_dim, hidden_dim//2),
            BiGRUModule(hidden_dim, hidden_dim//2),
            BiGRUModule(hidden_dim, hidden_dim//2)
        ])
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出维度
        self.out_dim = hidden_dim
        
    def forward(self, x):
        # x形状: (batch_size, channels, seq_len)
        # 调整形状为 (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # 层归一化 + 多头注意力 + 残差连接
        residual = x
        x = self.norm1(x)
        x = self.multihead_attn(x)
        x = x + residual  # 第一个残差连接
        
        # 层归一化 + 前馈层 + 残差连接
        residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = x + residual  # 第二个残差连接
        
        # 最后一个层归一化
        x = self.norm3(x)
        
        # BiGRU层处理 - 按照图中的连接方式
        residual = x  # 保存用于最后一个残差连接
        for i, bigru_layer in enumerate(self.bigru_layers):
            if i == 0:
                x = bigru_layer(x)
            else:
                # 对于后续的BiGRU层，我们添加前一层的输出
                x = bigru_layer(x) + x
        
        # 全局池化前添加最后的残差连接
        x = x + residual
        
        # 全局池化
        x = x.transpose(1, 2)  # 形状变为 (batch_size, channels, seq_len)
        x = self.global_pool(x).squeeze(-1)
        
        return x

# 局部特征提取模块
class LocalFeatureModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LocalFeatureModule, self).__init__()
        
        # 不同尺寸的卷积核 - 对应图中的3*dim, 5*dim, 7*dim卷积核
        self.conv3 = nn.Conv1d(16, hidden_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(16, hidden_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(16, hidden_dim, kernel_size=7, padding=3)
        
        # 1D ResNet层 - 对应图中的四层1DResNet
        self.resnet_layers = nn.ModuleList([
            ResNet1DBlock(hidden_dim * 3, hidden_dim * 2),
            ResNet1DBlock(hidden_dim * 2, hidden_dim * 2),
            ResNet1DBlock(hidden_dim * 2, hidden_dim),
            ResNet1DBlock(hidden_dim, hidden_dim)
        ])
        
        # BPA层 - 双线性池化注意力
        self.bpa = BPA(hidden_dim)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出维度
        self.out_dim = hidden_dim
        
    def forward(self, x):
        # 应用不同卷积核 (对应图中的3*dim, 5*dim, 7*dim卷积核)
        conv3_out = F.relu(self.conv3(x))
        conv5_out = F.relu(self.conv5(x))
        conv7_out = F.relu(self.conv7(x))
        
        # 合并不同卷积结果
        out = torch.cat([conv3_out, conv5_out, conv7_out], dim=1)
        
        # 通过ResNet层
        for resnet_layer in self.resnet_layers:
            out = resnet_layer(out)
        
        # 应用BPA (双线性池化注意力)
        out = self.bpa(out)
        
        # 全局池化并展平
        out = self.global_pool(out).squeeze(-1)
        
        return out

# 特征融合模块
class FeatureFusionModule(nn.Module):
    def __init__(self, local_dim, temporal_dim, output_dim):
        super(FeatureFusionModule, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(local_dim + temporal_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, local_features, temporal_features):
        # 连接两个特征
        combined = torch.cat([local_features, temporal_features], dim=1)
        return self.fusion(combined)

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 两层全连接网络（降维+升维）
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        # 全局平均池化
        y = self.avg_pool(x).view(b, c)
        # 生成通道权重向量
        y = self.fc(y).view(b, c, 1)
        # 加权
        return x * y.expand_as(x)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 确保kernel_size是奇数
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        # 空间维度的卷积操作
        self.conv1 = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 沿通道维度求平均
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 空间卷积
        avg_out = self.conv1(avg_out)
        # 生成空间权重矩阵
        weight = self.sigmoid(avg_out)
        # 加权
        return x * weight

# ACT-BiGRU模型中的残差块
class ACTResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACTResidualBlock, self).__init__()
        # 第一个卷积层：3x1卷积核
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 第二个卷积层：5x1卷积核
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接的调整层（如果输入输出通道不一致）
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
        self.relu2 = nn.ReLU(inplace=True)
        
        # 自适应特征增强：通道注意力和空间注意力的组合
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention(kernel_size=7)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用注意力机制
        out = self.ca(out)
        out = self.sa(out)
        
        out += residual
        out = self.relu2(out)
        
        return out

# ACT-BiGRU模型
class ACTBiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_dim=64, num_classes=10, use_cnn=False):
        super(ACTBiGRUModel, self).__init__()
        
        self.use_cnn = use_cnn
        
        if use_cnn:
            # 使用CNN处理RGB图像
            self.cnn_layers = nn.Sequential(
                # 第一个卷积块
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # 第二个卷积块
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # 第三个卷积块
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # 自适应池化到固定大小
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
            # 分类器
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            # 局部特征提取通道
            # 输入预处理
            self.preprocess = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True)
            )
            
            # 3个残差块（带注意力机制）
            self.res_blocks = nn.Sequential(
                ACTResidualBlock(64, 64),
                ACTResidualBlock(64, 128),
                ACTResidualBlock(128, 128)
            )
            
            # 全局池化
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            
            # 时序特征提取通道 - BiGRU
            self.bigru = nn.GRU(
                input_size=128,  # 从残差块输出的特征维度
                hidden_size=128, # 隐藏状态维度
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            
            # 特征融合层
            self.fusion = nn.Sequential(
                nn.Linear(128 + 256, 512),  # 局部特征(128) + BiGRU输出(256)
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # 分类器
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
            # 小波变换层
            self.wavelet_layer = nn.Sequential(
                # ... 小波变换实现
            )
            
            # 多尺度特征融合
            self.multi_scale_fusion = nn.ModuleList([
                nn.Conv1d(channels, channels, kernel_size=k) 
                for k in [3, 5, 7, 11, 15]
            ])
    
    def forward(self, x):
        if self.use_cnn:
            # CNN处理RGB图像
            features = self.cnn_layers(x)
            output = self.classifier(features)
            return output
        else:
            # 输入形状调整为(batch_size, 1, sequence_length)
            x = x.unsqueeze(1)
            
            # 局部特征提取
            x = self.preprocess(x)
            local_features = self.res_blocks(x)
            
            # 全局池化提取局部特征
            pooled_local = self.global_pool(local_features).squeeze(-1)  # [batch, channels]
            
            # 准备时序输入
            # 转换为 [batch, seq_len, channels] 用于GRU
            seq_features = local_features.transpose(1, 2)  # [batch, seq_len, channels]
            
            # BiGRU处理时序特征
            temporal_features, _ = self.bigru(seq_features)
            
            # 提取最后一个时间步的输出
            last_temporal = temporal_features[:, -1, :]  # [batch, 2*hidden_size]
            
            # 特征融合
            combined = torch.cat([pooled_local, last_temporal], dim=1)
            fused = self.fusion(combined)
            
            # 分类
            output = self.classifier(fused)
            
            return output

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

# 绘制混淆矩阵的函数
def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    # 打印分类报告
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    print("\n分类报告:")
    print(report)

# 主函数，用于演示如何使用这些类
def main():
    # 设置随机种子，确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 使用融合图像模式
    use_fusion = True
    target_size = (224, 224)  # 设置图像大小
    
    if use_fusion:
        # 创建图像转换
        train_transform = ImageTransforms(is_training=True)
        val_transform = ImageTransforms(is_training=False)
        
        # 创建数据集 - 确保传递正确的transform对象
        train_dataset = RCSDataset(data_dir='DATA_01', transform=train_transform, use_fusion=True, target_size=target_size)
        val_dataset = RCSDataset(data_dir='DATA_01', transform=val_transform, use_fusion=True, target_size=target_size)
        
        # 创建ACT-BiGRU模型 - 使用CNN模式
        model = ACTBiGRUModel(input_size=None, num_classes=10, use_cnn=True)
        print("使用CNN模式的ACT-BiGRU模型处理融合图像数据")
    else:
        # 使用原始特征向量模式
        transform = RCSTransform(normalize=True, augment=True)
        dataset = RCSDataset(data_dir='DATA_01', transform=transform, use_fusion=False)
        
        # 获取第一个样本来确定特征维度
        first_sample, _ = dataset[0]
        input_size = first_sample.shape[0]
        
        # 创建ACT-BiGRU模型 - 使用原始架构
        model = ACTBiGRUModel(input_size=input_size, num_classes=10, use_cnn=False)
        print(f"使用ACT-BiGRU模型处理原始特征数据（特征维度: {input_size}）")
    
    # 划分训练集和验证集
    if use_fusion:
        # 使用已经分好的数据集
        train_indices = list(range(len(train_dataset)))
        val_indices = list(range(len(val_dataset)))
    else:
        # 从一个数据集中划分
        train_indices, val_indices = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.labels
        )
    
    # 创建数据加载器
    if use_fusion:
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=2
        )
    else:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(
            dataset, batch_size=32, sampler=train_sampler, num_workers=2
        )
        val_loader = DataLoader(
            dataset, batch_size=32, sampler=val_sampler, num_workers=2
        )
    
    model = model.to(device)
    
    # 打印模型结构
    print(f"\n模型结构:\n{model}\n")
    
    # 创建训练器
    trainer = Trainer(model, device=device)
    
    # 训练模型
    epochs = 150
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
    
    # 绘制混淆矩阵
    print("正在生成混淆矩阵...")
    true_labels = []
    pred_labels = []
    
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 获取预测标签和真实标签
            pred = output.argmax(dim=1).cpu().numpy()
            true = target.argmax(dim=1).cpu().numpy()
            
            true_labels.extend(true)
            pred_labels.extend(pred)
    
    # 获取类别名称
    if use_fusion:
        class_names = [str(i) for i in range(10)]  # 如果没有具体名称，就使用数字
    else:
        class_names = [str(idx) for idx in sorted(dataset.label_map.keys())]
    
    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, pred_labels, class_names, save_path='confusion_matrix.png')

if __name__ == "__main__":
    main() 