import numpy as np
import torch
import torch.nn as nn
import os
import math
from torch_geometric.nn import GCNConv, global_add_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        # x 表示节点特征矩阵，edge_index 表示边的索引矩阵
        x, edge_index = data.x, data.edge_index
        # 第一层图卷积 + ReLU 激活
        x = self.conv1(x, edge_index).relu()
        # 第二层图卷积 + ReLU 激活
        x = self.conv2(x, edge_index).relu()
        return x

class Embedder(nn.Module):
    def __init__(self, d_input, d_model):
        super(Embedder, self).__init__()
        # 1D 卷积层：将输入特征维度映射到模型维度
        self.conv1d = nn.Conv1d(d_input, d_model, 1)
        # 批归一化，提高训练稳定性
        self.batch_norm = nn.BatchNorm1d(d_model)

    def forward(self, inputs):
        # 输入形状: [batch_size, d_input, seq_len]
        embeddings = self.conv1d(inputs)  # [batch_size, d_model, seq_len]
        embeddings = self.batch_norm(embeddings)  # 批归一化
        # 调整维度为 [batch_size, seq_len, d_model]
        return embeddings.permute(0, 2, 1)

class Pointer(nn.Module):
    def __init__(self, d_query, d_unit):
        super(Pointer, self).__init__()
        self.tanh = nn.Tanh()
        # 用于将 query 映射到与 refs 相同的维度
        self.w_l = nn.Linear(d_query, d_unit, bias=False)
        # 可学习参数 v，用于计算注意力得分
        self.v = nn.Parameter(torch.FloatTensor(d_unit), requires_grad=True)
        # 使用均匀分布初始化 v
        self.v.data.uniform_(-(1. / math.sqrt(d_unit)), 1. / math.sqrt(d_unit))

    def forward(self, refs, query, mask):
        # refs: [batch_size, num_refs, d_unit]
        # query: [batch_size, d_query]
        # mask: [batch_size, num_refs]，用于屏蔽已选择的项
        # 将 query 映射并扩展为与 refs 相加
        query_proj = self.w_l(query).unsqueeze(1)  # [batch_size, 1, d_unit]
        # 注意力得分：先 tanh 激活再与 v 相乘求和
        scores = torch.sum(self.v * self.tanh(refs + query_proj), dim=-1)  # [batch_size, num_refs]
        scores = 10. * self.tanh(scores)  # 增强区分度
        # 对已屏蔽项设为 -inf，避免被选中
        with torch.no_grad():
            scores[mask] = float('-inf')
        return scores  # 返回注意力得分，用于 softmax 或 top-k 选择

class Glimpse(nn.Module):
    def __init__(self, d_model, d_unit):
        super(Glimpse, self).__init__()
        self.tanh = nn.Tanh()
        # 用于提取每个位置的重要性
        self.conv1d = nn.Conv1d(d_model, d_unit, 1)
        # 可学习参数 v，用于计算注意力得分
        self.v = nn.Parameter(torch.FloatTensor(d_unit), requires_grad=True)
        self.v.data.uniform_(-(1. / math.sqrt(d_unit)), 1. / math.sqrt(d_unit))
        # 用 softmax 进行归一化得到注意力权重
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encs):
        # encs: [batch_size, seq_len, d_model]
        # 对输入做 1D 卷积并激活
        encoded = self.conv1d(encs.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, seq_len, d_unit]
        scores = torch.sum(self.v * self.tanh(encoded), dim=-1)  # [batch_size, seq_len]
        attention = self.softmax(scores)  # [batch_size, seq_len]
        # 根据注意力权重加权求和，得到 glimpse 向量
        glimpse = attention.unsqueeze(-1) * encs  # [batch_size, seq_len, d_model]
        glimpse = torch.sum(glimpse, dim=1)  # [batch_size, d_model]
        return glimpse  # 输出聚合后的上下文向量
