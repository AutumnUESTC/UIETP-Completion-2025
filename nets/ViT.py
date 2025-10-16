from copy import deepcopy

import math
import numpy as np
import torch

from torch import nn

# 构建图像编码模块
class Embeddings(nn.Module):
    '''
    对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
    backbone的输入input[2, 512, 64, 64]
    '''

    def __init__(self, img_size, in_channels):
        super(Embeddings, self).__init__()
        img_size = img_size  # 64
        patch_size = 2
        hidden_size = 512
        # 图像块数 (64/2)^2 = 1024
        n_patches = int((img_size / patch_size) * (img_size / patch_size))
        # 对图片进行卷积获取图片的块，并且将每一块映射成hidden_size维
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)

        # 设置可学习的位置编码信息
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        bs = x.shape[0]
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings  # 将图片块信息和对其位置信息进行相加
        embeddings = self.dropout(embeddings)
        return embeddings


# 构建self-Attention模块
class Attention(nn.Module):
    def __init__(self, vis, hidden_size=512):
        super(Attention, self).__init__()
        num_heads = 8
        self.vis = vis
        self.num_attention_heads = num_heads  # 8
        self.attention_head_size = int(hidden_size / self.num_attention_heads)  # 512/8=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 64*8=512

        self.query = nn.Linear(hidden_size, self.all_head_size)  # wm,512->512，Wq矩阵为（512,512）
        self.key = nn.Linear(hidden_size, self.all_head_size)  # wm,512->512,Wk矩阵为（512,512）
        self.value = nn.Linear(hidden_size, self.all_head_size)  # wm,512->512,Wv矩阵为（512,512）
        self.out = nn.Linear(hidden_size, hidden_size)  # wm,512->512
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)  # wm,(bs,65)+(2,256)=(bs,65,2,256)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,256,65,2)

    def forward(self, hidden_states):
        # hidden_states为：(bs,65,512)
        mixed_query_layer = self.query(hidden_states)  # wm,512->512
        mixed_key_layer = self.key(hidden_states)  # wm,512->512
        mixed_value_layer = self.value(hidden_states)  # wm,512->512

        query_layer = self.transpose_for_scores(mixed_query_layer)  # wm，(bs,256,65,2)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # 将q向量和k向量进行相乘（bs,12,197,197)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 将结果除以向量维数的开方
        attention_probs = self.softmax(attention_scores)  # 将得到的分数进行softmax,得到概率
        weights = attention_probs if self.vis else None  # wm,实际上就是权重
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # 将概率与内容向量相乘
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # wm,(bs,197)+(768,)=(bs,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


# 构建前向传播神经网络
# 两个全连接神经网络，中间加了激活函数
class Mlp(nn.Module):
    def __init__(self, hidden_size=512):
        super(Mlp, self).__init__()
        mlp_dim = 1024
        self.fc1 = nn.Linear(hidden_size, mlp_dim)  # wm,786->3072
        self.fc2 = nn.Linear(mlp_dim, hidden_size)  # wm,3072->786
        self.act_fn = torch.nn.functional.gelu  # wm,激活函数
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)  # wm,786->3072
        x = self.act_fn(x)  # 激活函数
        x = self.dropout(x)  # wm,丢弃
        x = self.fc2(x)  # wm3072->786
        x = self.dropout(x)
        return x


# 构建编码器的可重复利用的Block()模块：每一个block包含了self-attention模块和MLP模块
class Block(nn.Module):
    def __init__(self, vis, hidden_size=512):
        super(Block, self).__init__()
        self.hidden_size = hidden_size  # wm,512
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)  # wm，层归一化
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.ffn = Mlp(hidden_size)
        self.attn = Attention(vis, hidden_size)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h  # 残差结构

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h  # 残差结构
        return x, weights


# 构建Encoder模块，该模块实际上就是堆叠N个Block模块
class Encoder(nn.Module):

    def __init__(self, vis, in_channels=512):
        super(Encoder, self).__init__()
        num_layers = 12  # block模块的个数d
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(in_channels, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(vis, in_channels)
            self.layer.append(deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


# reshape from (B, n_patch, hidden) to (B, h, w, hidden)
class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Transformer(nn.Module):
    def __init__(self, out_channel):
        super(Transformer, self).__init__()
        self.embedding = Embeddings(img_size=64, in_channels=out_channel)
        self.encoder = Encoder(vis=True, in_channels=512)
        self.reconstruct = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))

    def forward(self, features):  # features cuda:1
        embedding_output = self.embedding(features)
        encoded, attn_weights = self.encoder(embedding_output)
        reconstruct = self.reconstruct(encoded)
        return reconstruct