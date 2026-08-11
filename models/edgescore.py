from torch._C import TensorType
import torch.nn as nn
from copy import deepcopy
import torch

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

def get_vector_and_coord(coords):
    _,C,N = coords.size()
    all_coords = coords.unsqueeze(2).repeat(1,1,N,1)
    temp = torch.arange(0,N).unsqueeze(0).unsqueeze(0).to('cuda')
    idx = torch.arange(0,N-1).unsqueeze(0).unsqueeze(0).repeat(1,N,1).to('cuda')
    for i in range(0,N):
        idx[0,i,:] = torch.cat([temp[0,0,:i], temp[0,0,i+1:]])
    idx = idx.unsqueeze(1).repeat(1,C,1,1)
    neighbor_coords = torch.gather(all_coords, dim=-1,index=idx)
    coords = coords.unsqueeze(-1).repeat(1,1,1,N-1)
    vectors = coords-neighbor_coords
    return torch.cat([vectors/2+neighbor_coords, vectors], dim=1)

def get_vector(coords):
    _,C,N = coords.size()
    all_coords = coords.unsqueeze(2).repeat(1,1,N,1)
    temp = torch.arange(0,N).unsqueeze(0).unsqueeze(0).to('cuda')
    idx = torch.arange(0,N-1).unsqueeze(0).unsqueeze(0).repeat(1,N,1).to('cuda')
    for i in range(0,N):
        idx[0,i,:] = torch.cat([temp[0,0,:i], temp[0,0,i+1:]])
    idx = idx.unsqueeze(1).repeat(1,C,1,1)
    neighbor_coords = torch.gather(all_coords, dim=-1,index=idx)
    coords = coords.unsqueeze(-1).repeat(1,1,1,N-1)
    return coords-neighbor_coords

class MyMultiHeadAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        return prob

class MyAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MyMultiHeadAttention(num_heads, feature_dim)

    def forward(self, x, source):
        prob = self.attn(x, source, source)
        return prob

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x*100))

class EdgeScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = AttentionalPropagation(32, 1)
        self.score_fna = nn.Sequential(
            nn.Linear(32, 1, bias=False),
            #nn.Linear(64, 1, bias=False)
            #nn.Softmax(dim=-1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(6, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
        )

    def regular_score(self,score):
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score

    def forward(self, src_vectors, tgt_vectors):
        N1 = src_vectors.size()[2]
        N2 = tgt_vectors.size()[2]
        """ src_vectors_em = self.pointnet(src_vectors)
        tgt_vectors_em = self.pointnet(tgt_vectors) """
        src_vectors = self.mlp(src_vectors.view(1,6,N1,N1-1).permute(0,2,3,1)).permute(0,3,1,2).view(1,32,N1*(N1-1))
        tgt_vectors = self.mlp(tgt_vectors.view(1,6,N2,N2-1).permute(0,2,3,1)).permute(0,3,1,2).view(1,32,N2*(N2-1))
        sc = tgt_vectors + self.attention(tgt_vectors, src_vectors)
        """ src_vectors = 
        tgt_vectors =  """
        """ prob = self.attention(tgt_vectors_em, src_vectors_em)
        prob = torch.max(prob, 3, keepdim=True)[0].view(1,1,N2,N2-1)
        sigmoid = nn.Sigmoid()
        prob = sigmoid(prob) """
        sc = sc.view(1,32,N2,N2-1)
        sc = self.score_fna(sc.permute(0,2,3,1)).permute(0,3,1,2)
        sigmoid = nn.Sigmoid()
        sc = sigmoid(sc)
        sc = torch.clamp(sc,min=0,max=1)
        sc = self.regular_score(sc)
        return sc