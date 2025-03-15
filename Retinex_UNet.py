import torch
import torch.nn as nn
from einops import rearrange
import math
import torch.nn.functional as F

class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(Cross_Attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = dim // num_heads

        self.query = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=False)
        self.key = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=False)
        self.value = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value=None):
        if value is None:
            value = key

        batch_size = query.size(0)
        h_q, w_q = query.size(2), query.size(3)
        h_k, w_k = key.size(2), key.size(3)

        # 处理query
        q = self.query(query)
        q = q.view(batch_size, self.num_heads, self.attention_head_size, h_q, w_q)
        q = q.permute(0, 1, 3, 4, 2).contiguous()  # (batch, heads, h, w, head_dim)
        q = q.view(batch_size, self.num_heads, h_q * w_q, self.attention_head_size)  # (batch, heads, h*w, head_dim)

        # 处理key
        k = self.key(key)
        k = k.view(batch_size, self.num_heads, self.attention_head_size, h_k, w_k)
        k = k.permute(0, 1, 3, 4, 2).contiguous()
        k = k.view(batch_size, self.num_heads, h_k * w_k, self.attention_head_size)

        # 处理value
        v = self.value(value)
        v = v.view(batch_size, self.num_heads, self.attention_head_size, h_k, w_k)
        v = v.permute(0, 1, 3, 4, 2).contiguous()
        v = v.view(batch_size, self.num_heads, h_k * w_k, self.attention_head_size)

        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # (batch, heads, h_q*w_q, h_k*w_k)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # 计算上下文向量
        context = torch.matmul(attention_probs, v)  # (batch, heads, h_q*w_q, head_dim)
        # 重组为空间维度
        context = context.view(batch_size, self.num_heads, h_q, w_q, self.attention_head_size)
        context = context.permute(0, 1, 4, 2, 3).contiguous()  # (batch, heads, head_dim, h, w)
        context = context.view(batch_size, self.num_heads * self.attention_head_size, h_q, w_q)  # (batch, dim, h, w)

        return context


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = self.relu(out)
        return out

class RetinexDecom(nn.Module):
    def __init__(self, channels):
        super(RetinexDecom, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.blocks0 = nn.Sequential(*[ResidualBlock(channels, channels) for _ in range(2)])
        self.illum_conv = nn.Conv2d(1, channels, kernel_size=1)  # 新增卷积处理通道数
        self.cross_attention = Cross_Attention(dim=channels, num_heads=8)
        self.self_attention = Self_Attention(dim=channels, num_heads=8, bias=True)
        self.conv0_1 = nn.Sequential(*[ResidualBlock(channels, channels), nn.Conv2d(channels, channels, kernel_size=3, padding=1)])

    def forward(self, x):
        # 使用高斯滤波估计光照
        init_illumination = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        init_illumination = F.avg_pool2d(init_illumination, kernel_size=3, stride=1, padding=1)

        init_reflectance = x / (init_illumination + 1e-8)  # 避免除零
        illumination = self.self_attention(init_illumination)
        illumination = self.conv0_1(illumination)

        Reflectance = self.blocks0(self.conv0(init_reflectance))
        
        # 处理init_illumination的通道数
        Reflectance_final = self.cross_attention(Reflectance, illumination)
        Reflectance_final = self.conv0_1(Reflectance_final)

        return Reflectance_final, illumination


class RetinexUNet(nn.Module):
    def __init__(self):
        super(RetinexUNet, self).__init__()

        # 编码器部分
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Retinex 分解模块
        self.retinex_decom = RetinexDecom(512)

        # 解码器部分
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # 修改输入通道数为512
        self.dec4 = self.conv_block(1024, 512)  # 1024 = 512 (from upconv4) + 512 (from enc4)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)  # 512 = 256 (from upconv3) + 256 (from enc3)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)  # 256 = 128 (from upconv2) + 128 (from enc2)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)  # 128 = 64 (from upconv1) + 64 (from enc1)

        # 最终输出层
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Retinex 分解
        Reflectance, Illumination = self.retinex_decom(enc4)
        dec4 = Reflectance * Illumination
        
        # 解码器
        dec4 = torch.cat((dec4, enc4), dim=1)  # 拼接后通道数为1024
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # 最终输出
        output = self.final_conv(dec1)
        return output

if __name__ == "__main__":
    model = RetinexUNet()
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(output.shape)  # 应输出 torch.Size([1, 3, 256, 256])