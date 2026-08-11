import torch
import torch.nn as nn
from copy import deepcopy
from models.gcn import MultiHeadedAttention, MLP

def channel_wise_attention(query, key, value):
    dim = query.shape[1]
    scores = torch.matmul(query.permute(0,3,1,2), key.permute(0,3,2,1))
    #scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    #torch.einsum('bhnm,bdhm->bdhn', prob, value), prob
    res = torch.sum(prob*value.permute(0,3,2,1).repeat(1,1,value.size()[1], 1), dim=-1, keepdim=True)
    return res.permute(0,1,3,2), prob

class ChannelAttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        # change it to sum
        # return self.mlp(torch.cat([x, message], dim=1))
        return self.mlp(x + message)