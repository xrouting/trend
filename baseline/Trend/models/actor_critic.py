import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from models.utils import Embedder, Pointer, Glimpse,GCN
from models.self_attn import Encoder

class Actor(nn.Module):
    def __init__(self, device):
        super(Actor, self).__init__()
        #self.degree = degree
        self.device = device

        # embedder args
        self.d_input = 10
        self.d_middle = 64
        self.d_model = 128
        
        #xroute
        self.GCNembeder = GCN(in_channels=self.d_input, hidden_channels=self.d_middle, out_channels=self.d_model)
        #A2C test
        self.embedder = Embedder(self.d_input, self.d_model)

        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        # feedforward layer inner
        self.d_inner = 512

        self.encoder = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model, self.d_inner)

        # decoder args
        self.d_unit = 256
        self.d_query = 360
        self.conv1d_r = nn.Conv1d(self.d_model, self.d_unit, 1)
        self.conv1d_x = nn.Conv1d(self.d_model, self.d_unit, 1)
        self.conv1d_y = nn.Conv1d(self.d_model, self.d_unit, 1)

        self.start_ptr = Pointer(self.d_query, self.d_unit)
        self.q_l1 = nn.Linear(self.d_model, self.d_query, bias=False)
        self.relu = nn.ReLU()
        self.ctx_linear = nn.Linear(self.d_query, self.d_query, bias=False)

        self.ptr1 = Pointer(self.d_query, self.d_unit)

        self.to(device)
        self.train()

    def forward(self, observation, deterministic=False):
        # encode encode encode            
        #observation 
        num_net = observation.x.shape[0]
        embedings = self.GCNembeder(observation).unsqueeze(0)
        encodings = self.encoder(embedings, None).permute(0, 2, 1)
        enc_r = self.conv1d_r(encodings).permute(0, 2, 1)
        #enc_x = self.conv1d_x(encodings).permute(0, 2, 1)
        #enc_y = self.conv1d_y(encodings).permute(0, 2, 1)
        #1.卷积特征提取
        encodings = encodings.permute(0, 2, 1)
        #enc_xy = torch.cat([enc_x, enc_y], 1)

        batch_size = encodings.size()[0]
        #2.跟踪已经访问过的序列位置，初始时所有位置均未访问。
        visited = torch.zeros([batch_size, num_net], dtype=torch.bool).to(self.device)
        #3.indexed用于记录解码过程中的选择序列。
        #4.log_probs用于记录解码过程中每一步的对数概率。
        indexes, log_probs = [], []


        # initial_idx = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        # 初始查询向量为全0，后续可能需要修改
        start_logits = self.start_ptr(enc_r,
            torch.zeros([batch_size, self.d_query], dtype=torch.float).to(self.device), visited)
        
        #5.将注意力分数转化为一个类别分布并选择。
        distr = Categorical(logits=start_logits)
        #6.如果 deterministic=True，选择分数最高的索引。如果为 False，则根据分布采样索引。
        if deterministic:
            _, start_idx = torch.max(start_logits, -1)
        else:
            start_idx = distr.sample()

        #7.选择第一个net
        indexes.append(start_idx)
        visited.scatter_(1, start_idx.unsqueeze(-1), True)
        log_probs.append(distr.log_prob(start_idx))

        q1 = encodings[torch.arange(batch_size), start_idx]

        context = torch.zeros([batch_size, self.d_query]).to(self.device)

        for step in range(num_net - 1):
            #8.上一轮选定的编码 q1 通过全连接层self.q_l1 转化为查询向量
            residual = self.q_l1(q1) 
            #9.更新上下文 并构造查询向量first_query
            context = torch.max(context, self.ctx_linear(self.relu(residual)))
            first_q = residual + context
            first_query = self.relu(first_q)
            #10.计算注意力分数并根据是否确定性选择下一个net
            logits = self.ptr1(enc_r, first_query, visited)
            distr = Categorical(logits=logits)
            if deterministic:
                _, first_idx = torch.max(logits, -1)
            else:
                first_idx = distr.sample()

            indexes.append(first_idx)
            log_probs.append(distr.log_prob(first_idx))
            visited.scatter_(1, first_idx.unsqueeze(-1), True)
            q1 = encodings[torch.arange(encodings.size(0)), first_idx]

        indexes = torch.stack(indexes, -1)
        log_probs = sum(log_probs)

        return indexes, log_probs

class Critic(nn.Module):
    def __init__(self,  device):
        super(Critic, self).__init__()
        #self.degree = degree
        self.device = device

        # embedder args
        self.d_input = 10
        self.d_middle = 64
        self.d_model = 128

        # encoder args
        self.num_stacks = 3
        self.num_heads = 16
        self.d_k = 16
        self.d_v = 16
        self.d_inner = 512
        self.d_unit = 256

        self.crit_embedder = Embedder(self.d_input, self.d_model)
        self.crit_GCNembedder = GCN(in_channels=self.d_input, hidden_channels=self.d_middle, out_channels=self.d_model)
        
        
        self.crit_encoder = Encoder(self.num_stacks, self.num_heads, self.d_k, self.d_v, self.d_model, self.d_inner)
        self.glimpse = Glimpse(self.d_model, self.d_unit)
        self.critic_l1 = nn.Linear(self.d_model, self.d_unit)
        self.critic_l2 = nn.Linear(self.d_unit, 1)
        self.relu = nn.ReLU()

        self.to(device)
        self.train()

    def forward(self, observation, deterministic=False):
        inputs_tensor = self.crit_GCNembedder(observation).unsqueeze(0)
        critic_encode = self.crit_encoder(inputs_tensor, None)
        glimpse = self.glimpse(critic_encode)
        critic_inner = self.relu(self.critic_l1(glimpse))
        predictions = self.relu(self.critic_l2(critic_inner)).squeeze(-1)

        return predictions