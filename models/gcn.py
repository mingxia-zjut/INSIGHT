import torch
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
import torch.utils.checkpoint as checkpoint
from lib.utils import square_distance
import math as m
import timeit
from models.randla import *
import math
from models.edgescore import EdgeScore, get_vector_and_coord
#from models.DGCNN_PAConv import PAConv
from models.geotransformer.utils.point_cloud_utils import pairwise_distance
from models.geotransformer.modules.attention.positional_embedding import SinusoidalPositionalEmbedding
import numpy as np

'''
for point in tgt_cord:
    for pattern in src_patterns:
        if pattern_similar(tgt_cord, tgt_point_to_center[point], pattern):

            tgt_pattern = gen_pattern(tgt_cord, tgt_point_to_center[point], pattern)
            aggregate_feature_with_pattern(point, pattern, tgt_center_to_point)

        else:

            point is unavailable(that is, stay as usual or aggregate feature from K nearest neighbor?)
'''

def distance(p1, p2):
    return m.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2))

# TODO: 查看pattern匹配度 return sim, tgt_pattern
# cents: tuple; cord: tuple; pattern: list; th: flaot;
def pattern_similarity(cents, cent, pattern, th):
    #print(cent)
    """ if cent[0] == -1.872783209360728:
        print(1) """
    tgt_pattern = []
    for vector in pattern:
        right_pos = tuple([cent[0]+vector[0], cent[1]+vector[1], cent[2]+vector[2]])
        for point in cents:
            if point == cent:
                continue
            if distance(point, right_pos) <= th:
                tgt_pattern.append(point)
    if len(tgt_pattern) == len(pattern):
        return True, tgt_pattern
    return False, tgt_pattern

memory = {}

# TODO: 根据元数据产生一个点的特征
def get_one_point_feature_with_meta_data(coords, feats, cord, feat, src_patterns, tgt_point2center, tgt_center2point, th):
    C = feat.size()[0]
    index = []
    OK = False   
    for key in src_patterns:
        c = [float(num) for num in cord]
        if tgt_point2center[tuple(c)] in memory:
            OK = True
            index = list(memory[tgt_point2center[tuple(c)]])
            break
        else:
            cents = [cent for cent in tgt_center2point]
            OK, tgt_pattern = pattern_similarity(cents, tgt_point2center[tuple(c)], src_patterns[key], th)
            if OK:
                for tgt_cent in tgt_pattern:
                    for p1 in tgt_center2point[tgt_cent]:
                        for i in range(coords.size()[2]):
                            if tuple([float(coords[0,:,i][0]), float(coords[0,:,i][1]), float(coords[0,:,i][2])]) == p1:
                                index.append(i)
                memory[tgt_point2center[tuple(c)]] = tuple(index)
                break
    
    if OK:
        dist = square_distance(cord.unsqueeze(0).unsqueeze(-1).transpose(1,2), coords[:,:,index].transpose(1,2))
        idx = dist.topk(k=10+1, dim=-1, largest=False, sorted=True)[1] #[B, C, N, K]
        idx = idx[:,:,1:]  #[B, N, K]

        idx = idx.unsqueeze(1).repeat(1,C,1,1)
        all_feats = feats[:,:,index].unsqueeze(2).repeat(1, 1, 1, 1)  # [B, C, N, N]

        neighbor_feat = torch.gather(all_feats, dim=-1,index=idx) #[B, C, N, K]

        # concatenate the features with centroid
        feat = feat.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,10)

        feat_cat = torch.cat((feat, neighbor_feat-feat),dim=1)
    else:
        #feat_cat = torch.zeros([1, 2*C, 1, 10]).to(torch.device('cuda'))
        dist = square_distance(cord.unsqueeze(0).unsqueeze(-1).transpose(1,2), coords.transpose(1,2))
        idx = dist.topk(k=10+1, dim=-1, largest=False, sorted=True)[1] #[B, C, N, K]
        idx = idx[:,:,1:]  #[B, N, K]

        idx = idx.unsqueeze(1).repeat(1,C,1,1)
        all_feats = feats.unsqueeze(2).repeat(1, 1, 1, 1)  # [B, C, N, N]

        neighbor_feat = torch.gather(all_feats, dim=-1,index=idx) #[B, C, N, K]

        # concatenate the features with centroid
        feat = feat.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,10)

        feat_cat = torch.cat((feat, neighbor_feat-feat),dim=1)
    return feat_cat

def get_tgt_graph_feature(coords, feats, src_patterns, tgt_point2center, tgt_center2point, th):
    cord = coords[0,:,0]
    feat = feats[0,:,0]
    feat_cat = get_one_point_feature_with_meta_data(coords, feats, cord, feat, src_patterns, tgt_point2center, tgt_center2point, th)
    sum = 0
    for i in range(1, coords.size()[2]):
        cord = coords[0,:,i]
        feat = feats[0,:,i]
        start=timeit.default_timer()
        feat_cat = torch.cat([feat_cat, get_one_point_feature_with_meta_data(coords, feats, cord, feat, src_patterns, tgt_point2center, tgt_center2point, th)],dim=2)
        #中间写上代码块
        end=timeit.default_timer()
        if end-start > 0.1:
            sum += 1
    print('Running time: %s Seconds'%(end-start), sum)

    # init a tensor, concatenate the feature to this tensor and return
    return feat_cat

def get_tgt_graph_feature_per_center(coords, feats, src_patterns, tgt_center2point, tgt_index, th):
    B, C, N = feats.size()
    features = torch.zeros([B, C, 1, 10]).to(torch.device('cuda'))
    neighbors = torch.zeros([B, C, 1, 10]).to(torch.device('cuda'))
    all_feats = feats.unsqueeze(2).repeat(1, 1, N, 1)
    for center, point_index in tgt_center2point.items():
        flag = False
        for src_center, pattern in src_patterns.items():
            tgt_centers = [center for center in tgt_center2point]
            OK, centers = pattern_similarity(tgt_centers, center, pattern, th)
            if OK:
                index = []
                #print("1:{}".format(torch.cuda.memory_allocated(0)))
                [index.extend(list(tgt_center2point[center])) for center in centers]
                #print("1:{}".format(torch.cuda.memory_allocated(0)))
                dist = square_distance(coords[:,:,point_index].transpose(1,2), coords[:,:,index].transpose(1,2))
                idx = dist.topk(k=10+1, dim=-1, largest=False, sorted=True)[1] #[B, C, N, K]
                idx = idx[:,:,1:]  #[B, N, K]

                idx = idx.unsqueeze(1).repeat(1,C,1,1) #[B, C, N, K]
                
                all_feats_with_index = feats[:,:,index].unsqueeze(2).repeat(1, 1, len(point_index), 1)  # [B, C, N, N] 
                # neighbor_feat[i,j,k,n] = all_feats_with_index[i,j,k,idx[i,j,k,n]]
                neighbor_feat = torch.gather(all_feats_with_index, dim=-1,index=idx) #[B, C, N, K]

                # concatenate the features with centroid
                feat = feats[:,:,point_index].unsqueeze(-1).repeat(1,1,1,10)

                # feat_cat = torch.cat((feat, neighbor_feat-feat),dim=1)
                features = torch.cat((features, feat), dim=2)
                neighbors = torch.cat((neighbors, neighbor_feat), dim=2)
                flag = True
                break
        if flag is False:
            dist = square_distance(coords[:,:,point_index].transpose(1,2), coords.transpose(1,2))
            idx = dist.topk(k=10+1, dim=-1, largest=False, sorted=True)[1] #[B, C, N, K]
            idx = idx[:,:,1:]  #[B, N, K]

            idx = idx.unsqueeze(1).repeat(1,C,1,1) #[B, C, N, K]
              

            neighbor_feat = torch.gather(all_feats, dim=-1,index=idx) #[B, C, N, K]

            # concatenate the features with centroid
            feat = feats[:,:,point_index].unsqueeze(-1).repeat(1,1,1,10)

            #feat_cat = torch.cat((feat, neighbor_feat-feat),dim=1)
            features = torch.cat((features, feat), dim=2)
            neighbors = torch.cat((neighbors, neighbor_feat), dim=2)

    
    feat_cat = torch.cat((features[:,:,1:,:][:,:,tgt_index,:], (neighbors-features)[:,:,1:,:][:,:,tgt_index,:]),dim=1)

    return feat_cat

def get_graph_feature(coords, feats, pos_embeddings, k=10, is_em=1):
    """
    Apply KNN search based on coordinates, then concatenate the features to the centroid features
    Input:
        X:          [B, 3, N]
        feats:      [B, C, N]
        features like:
        f11    f21   ..   fN1
        f12    f22   ..   fN2
        ..     ..    ..   fN3
        f1256  f2256 ..   fN256
    Return:
        feats_cat:  [B, 2C, N, k]
    """
    # apply KNN search to build neighborhood
    B, C, N = feats.size()
    dist = square_distance(coords.transpose(1,2), coords.transpose(1,2))
    idx = dist.topk(k=k+1, dim=-1, largest=False, sorted=True)[1]  #[B, N, K+1], here we ignore the smallest element as it's the query itself  
    
    idx = idx[:,:,1:]  #[B, N, K]

    idx = idx.unsqueeze(1).repeat(1,C,1,1) #[B, C, N, K]
    all_feats = feats.unsqueeze(2).repeat(1, 1, N, 1)  # [B, C, N, N]

    neighbor_feats = torch.gather(all_feats, dim=-1,index=idx) #[B, C, N, K]

    # concatenate the features with centroid
    feats = feats.unsqueeze(-1).repeat(1,1,1,k)

    if is_em == 1:
        feats_cat = torch.cat((feats, neighbor_feats-feats + torch.gather(pos_embeddings.permute(0,3,1,2), dim=-1,index=idx)),dim=1)
    else:
        feats_cat = torch.cat((feats, neighbor_feats-feats),dim=1)

    return feats_cat


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

def get_tgt_graph_feature_with_attention_score(coords, feats, k=10):
    """
    Apply KNN search based on coordinates, then concatenate the features to the centroid features
    Input:
        X:          [B, 3, N]
        feats:      [B, C, N]
        features like:
        f11    f21   ..   fN1
        f12    f22   ..   fN2
        ..     ..    ..   fN3
        f1256  f2256 ..   fN256
    Return:
        feats_cat:  [B, 2C, N, k]
    """
    # apply KNN search to build neighborhood
    B, C, N = feats.size()
    dist = square_distance(coords.transpose(1,2), coords.transpose(1,2))

    idx = dist.topk(k=k+1, dim=-1, largest=False, sorted=True)[1]  #[B, N, K+1], here we ignore the smallest element as it's the query itself  
    idx = idx[:,:,1:]  #[B, N, K]

    idx = idx.unsqueeze(1).repeat(1,C,1,1) #[B, C, N, K]
    all_feats = feats.unsqueeze(2).repeat(1, 1, N, 1)  # [B, C, N, N]

    neighbor_feats = torch.gather(all_feats, dim=-1,index=idx) #[B, C, N, K]

    # concatenate the features with centroid
    feats = feats.unsqueeze(-1).repeat(1,1,1,k)

    feats_cat = torch.cat((feats, neighbor_feats-feats),dim=1)

    return feats_cat

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

class MySigmoid(nn.Module):
    def __init__(self):
        super(MySigmoid, self).__init__() 
        self.t = nn.Parameter(torch.Tensor([0]))
    def forward(self, x):
        return 1.0 /(1.0 + torch.exp(-(x)*100))

def get_lable(tgt_vector, src_vector):
    tgt_len = torch.sqrt(torch.sum(torch.mul(tgt_vector[0,:].permute(1,0),tgt_vector[0,:].permute(1,0)), dim=1)).unsqueeze(-1)
    src_len = torch.sqrt(torch.sum(torch.mul(src_vector[0,:].permute(1,0),src_vector[0,:].permute(1,0)), dim=1)).unsqueeze(0)
    distance_mul = torch.matmul(tgt_len, src_len)
    dot_product_res = torch.matmul(tgt_vector[0,:].permute(1,0), src_vector[0,:])
    angle = torch.acos(dot_product_res / distance_mul)
    dis_diff = torch.abs(tgt_len.repeat(1, src_len.size()[1]) - src_len.repeat(tgt_len.size()[0], 1))
    angle = (angle < (3/180)*math.pi).float()
    dis_diff = (dis_diff < 0.05).float()
    gt_labels = (torch.sum(angle*dis_diff, dim=1) > 0).float()
    dis_limit = (tgt_len > 0.13).float()
    
    return gt_labels * dis_limit.view(dis_limit.size()[0])


class SelfAttention(nn.Module):
    def __init__(self,feature_dim,k=10):
        super(SelfAttention, self).__init__() 
        self.conv1 = nn.Conv2d(feature_dim*2, feature_dim, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(feature_dim)
        
        self.conv2 = nn.Conv2d(feature_dim*2, feature_dim * 2, kernel_size=1, bias=False)
        self.in2 = nn.InstanceNorm2d(feature_dim * 2)

        self.conv3 = nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(feature_dim)
        

        """ self.score_fna = nn.Sequential(
            nn.Linear(3, 1, bias=False),
            #nn.Softmax(dim=-1)
        ) """

        """ self.proj_gnn = nn.Conv1d(3,3,kernel_size=1, bias=True)
        self.proj_score = nn.Conv1d(3,1,kernel_size=1,bias=True) """

        """ self.score_fnb = nn.Sequential(
            nn.Linear(1024, 1, bias=False),
            #nn.Softmax(dim=-1)
        ) """

        """ self.mlp1 = SharedMLP(256, 256, bn=False, activation_fn=nn.ReLU())
        self.mlp2 = SharedMLP(512, 512, bn=False, activation_fn=nn.ReLU()) """

        self.k = k

        self.attention = AttentionalPropagation(3, 1)

        self.t = nn.Parameter(torch.Tensor([0.5]))

        self.edgescore = EdgeScore()
        
        """ self.sigmoid = MySigmoid() """

    def regular_score(self,score):
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score

    """ def sigmoid(self, x):
        return 1.0 /(1.0 + torch.exp(-(x)*100)) """
    #原本is_src=1
    # TODO: 元数据要传进去
    def forward(self, coords, features, pos_embeddings, src_patterns, tgt_center2point, tgt_index, max_pool_x=None, src_vectors=None, tgt_vectors=None, is_src=1, is_em=1):
        """
        Here we take coordinats and features, feature aggregation are guided by coordinates
        Input: 
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        """
        # 根据map来聚合特征
        # B: batch size : 1
        # C: feature size
        # N: number of point
        B, C, N = features.size()

        x0 = features.unsqueeze(-1)  #[B, C, N, 1]
        sc = None
        if is_src == 0:
            if max_pool_x != None:
                max_pool_x = max_pool_x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, N, (N-1))
            """ sc = src_vectors + self.attention(src_vectors, tgt_vectors)
            #sc = self.proj_gnn(sc)
            #sc = self.proj_score(sc)
            sc = sc.view(1,3,N,N-1)
            #sc = self.attention(src_vectors, tgt_vectors)
            sc = self.score_fna(sc.permute(0,2,3,1)).permute(0,3,1,2)
            sigmoid = nn.Sigmoid()
            sc = sigmoid(sc)
            sc = torch.clamp(sc,min=0,max=1)
            sc = self.regular_score(sc)
            #sc = (sc > self.t).float()
            #sc[sc>self.t] = 1
            #sc[sc<=self.t] = 0 """
            sc = self.edgescore(tgt_vectors, src_vectors)
            #sc = (sc > 0.9).float()

        if is_src:
            x1 = get_graph_feature(coords, x0.squeeze(-1), pos_embeddings, self.k, is_em)
            x1 = F.leaky_relu(self.in1(self.conv1(x1)), negative_slope=0.2)
            x1 = x1.max(dim=-1,keepdim=True)[0]
        else:
            """ #x1 = get_tgt_graph_feature(coords, features, src_patterns, tgt_point2center, tgt_center2point, tgt_index, 0.1)
            #x1 = get_tgt_graph_feature_per_center(coords, features, src_patterns, tgt_center2point, tgt_index, 0.15)
            x1 = get_graph_feature(coords, x0.squeeze(-1), (N-1))
            #x1 = F.leaky_relu(self.in1(self.conv1(x1)), negative_slope=0.2)
            x1 = self.conv1(x1)
            x1_sc = self.score_fna(torch.cat((max_pool_x, x1),dim=1).permute(0,2,3,1)).permute(0,3,1,2)
            sigmoid = nn.Sigmoid()
            x1_sc = torch.clamp(sigmoid(x1_sc),min=0,max=1)
            x1_sc = self.regular_score(x1_sc)
            x1 = x1_sc * x1
            x1 = self.in1(x1)
            x1 = F.leaky_relu(x1, negative_slope=0.2)
            x1 = x1.max(dim=-1,keepdim=True)[0]
            #x1 = self.mlp1(x1)
            #x1 = x1.max(dim=-1,keepdim=True)[0] """

            x1 = get_graph_feature(coords, x0.squeeze(-1), pos_embeddings, (N-1),is_em)
            x1 = self.conv1(x1)
            x1 = sc * x1
            x1 = F.leaky_relu(self.in1(x1), negative_slope=0.2)
            x1 = x1.max(dim=-1,keepdim=True)[0]

        if is_src:
            x2 = get_graph_feature(coords, x1.squeeze(-1), pos_embeddings, self.k,is_em)
            x2 = F.leaky_relu(self.in2(self.conv2(x2)), negative_slope=0.2)
            x2 = x2.max(dim=-1, keepdim=True)[0]
        else:
            """ #x2 = get_tgt_graph_feature(coords, features, src_patterns, tgt_point2center, tgt_center2point, tgt_index, 0.1)
            #x2 = get_tgt_graph_feature_per_center(coords, features, src_patterns, tgt_center2point, tgt_index, 0.15)
            x2 = get_graph_feature(coords, x1.squeeze(-1), (N-1))
            #x2 = F.leaky_relu(self.in2(self.conv2(x2)), negative_slope=0.2)
            x2 = self.conv2(x2)
            #x2 = self.score_fn2(torch.cat((max_pool_x, x2),dim=1).permute(0,2,3,1)).permute(0,3,1,2) * x2
            x2_sc = self.score_fnb(torch.cat((max_pool_x, x2),dim=1).permute(0,2,3,1)).permute(0,3,1,2)
            sigmoid = nn.Sigmoid()
            x2_sc = torch.clamp(sigmoid(x2_sc),min=0,max=1)
            x2_sc = self.regular_score(x2_sc)
            x2 = x2_sc * x2
            x2 = self.in2(x2)
            x2 = F.leaky_relu(x2, negative_slope=0.2)
            x2 = x2.max(dim=-1, keepdim=True)[0]
            #x2 = self.mlp2(x2)
            #x2 = x2.max(dim=-1, keepdim=True)[0] """

            x2 = get_graph_feature(coords, x1.squeeze(-1), pos_embeddings, (N-1),is_em)
            x2 = self.conv2(x2)
            x2 = sc * x2
            x2 = F.leaky_relu(self.in2(x2), negative_slope=0.2)
            x2 = x2.max(dim=-1,keepdim=True)[0]

        
        

        x3 = torch.cat((x0,x1,x2),dim=1)
        x3 = F.leaky_relu(self.in3(self.conv3(x3)), negative_slope=0.2).view(B, -1, N)

        return x3


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


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


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
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalPropagation_1(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([message], dim=1))

""" def get_graph_matrix(coords):
    B,C,N = coords.size()
    dist = square_distance(coords.transpose(1,2), coords.transpose(1,2))
    adj_matrix = (dist < 2.0).float()-torch.eye(N).to('cuda')
    degree_matrix = torch.sum(adj_matrix, dim=-1, keepdim=True) * torch.eye(N).to('cuda')
    L = degree_matrix - adj_matrix
    #e, v = torch.eig(L.squeeze(0), eigenvectors=True)
    #e, v = torch.symeig(L.squeeze(0), eigenvectors=True)
    vals, vecs = eigen.eigs(np.asarray(L.squeeze(0).to('cpu')), k=1, which='SR')
    #X, LU = torch.solve(torch.zeros([N,1]), L.squeeze(0)-torch.eye(N)*e[2][0])
    return torch.tensor(vecs[:,0]).float().to('cuda') """

def get_graph_matrix(coords):
    B,C,N = coords.size()
    dist = square_distance(coords.transpose(1,2), coords.transpose(1,2))
    adj_matrix = (dist < 2.0).float()-torch.eye(N).to('cuda')
    degree_matrix = torch.sum(adj_matrix, dim=-1, keepdim=True) * torch.eye(N).to('cuda')
    L = degree_matrix - adj_matrix
    e, v = torch.eig(L.squeeze(0), eigenvectors=True)
    return v[:,0]

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel,reduction = 16):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        """ t = int(abs((m.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid() """
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # self.fc[0].weight.data.fill_(1)
        # self.fc[2].weight.data.fill_(1)


    def forward(self, u0, features):
        B,C,N = features.size()
        x = features * u0
        y = torch.sum(x, dim=-1)
        # Two different branches of ECA module
        """ y = self.conv(y.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
        y = self.sigmoid(y) """
        y = self.fc(y)
        return features * y.unsqueeze(-1)

from models.graph import GetGraph, GetLaplacian

def sel_freq(features, u, sel):
    B,C,N = features.size()
    if C % len(sel) != 0:
        print('top_k')
        return
    num = C // len(sel)
    res = u[:,sel[0]].view(1,1,N).repeat(1,num,1)
    for i in range(1, len(sel)):
        res = torch.cat([res, u[:,sel[i]].view(1,1,N).repeat(1,num,1)], dim=1)
    return res

import xmlrpc.client
import pickle

def hex_to_fea(fea_hex):
        fea_bytes = bytes.fromhex(fea_hex)
        feature=pickle.loads(fea_bytes)
        return feature
    
def feature_pickle_base(feature):
    fea_pik=pickle.dumps(feature)
    fea_hex =fea_pik.hex() # tensor 序列化成bytes再转成hex字符串（16进制）
    return fea_hex

class GCN(nn.Module):
    """
        Alternate between self-attention and cross-attention
        Input:
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        """
    def __init__(self, num_head: int, feature_dim: int, k: int, layer_names: list, fre, config):
        super().__init__()
        # self.server = xmlrpc.client.ServerProxy("http://localhost:8000")
        self.layers=[]
        self.fre = fre
        self.fre1 = config.f1
        self.fre2 = config.f2
        self.mode = config.mode
        for atten_type in layer_names:
            if atten_type == 'cross':
                self.layers.append(AttentionalPropagation(feature_dim,num_head))
            elif atten_type == 'self':
                self.layers.append(SelfAttention(feature_dim, k))
        self.layers = nn.ModuleList(self.layers)
        self.names = layer_names
        self.channelattention = MultiSpectralAttentionLayer(256)
        self.getGraph = GetGraph()
        self.getLaplacian = GetLaplacian(normalize=True)
        self.embedding = SinusoidalPositionalEmbedding(256)
        self.sigma_d = 0.2
        self.angle_k = 3
        self.sigma_a = 15
        self.factor_a = 180. / (self.sigma_a * np.pi)
        self.rde_proj = nn.Linear(256, 256)
        self.rae_proj = nn.Linear(256, 256)

    def get_geometric_structure_embeddings(self, points):
        with torch.no_grad():
            batch_size, num_point, _ = points.shape

            dist_map = torch.sqrt(pairwise_distance(points, points, clamp=True))  # (B, N, N)
            square_distance(points, points)
            rde_indices = dist_map / self.sigma_d

            knn_indices = dist_map.topk(k=self.angle_k + 1, dim=2, largest=False)[1]  # (B, N, k)
            knn_indices = knn_indices[:, :, 1:]
            knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, self.angle_k, 3)  # (B, N, k, 3)
            expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
            knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
            ref_vectors = knn_points - points.unsqueeze(2)  # (B, N, k, 3)
            anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
            ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, self.angle_k, 3)  # (B, N, N, k, 3)
            anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, self.angle_k, 3)  # (B, N, N, k, 3)
            sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, k)
            cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, k)
            angles = torch.atan2(sin_values, cos_values)  # (B, N, N, k)
            rae_indices = angles * self.factor_a

        rde = self.embedding(rde_indices)  # (B, N, N, C)
        rde = self.rde_proj(rde)  # (B, N, N, C)

        rae = self.embedding(rae_indices)  # (B, N, N, k, C)
        rae = self.rae_proj(rae)  # (B, N, N, k, C)
        rae = rae.max(dim=3)[0]  # (B, N, N, C)

        gse = rde + rae  # (B, N, N, C)

        return gse

    # TODO: 参数要传入元数据
    def forward(self, coords0, coords1, desc0, desc1, base, name0, name1, src_patterns, tgt_center2point, tgt_index, max_pool_x, u0_0=None, u0_1=None):
        src_embeddings = self.get_geometric_structure_embeddings(coords0.transpose(1,2))
        tgt_embeddings = self.get_geometric_structure_embeddings(coords1.transpose(1,2))

        if(self.mode == 'test'):
            G0 = self.getGraph(coords0.permute(0,2,1))

            L0 = self.getLaplacian(G0)
            _, u0_0 = torch.symeig(L0.squeeze(0), eigenvectors=True)


            G1 = self.getGraph(coords1.permute(0,2,1))

            L1 = self.getLaplacian(G1)
            _, u0_1 = torch.symeig(L1.squeeze(0), eigenvectors=True)

            len0 = int((coords0.size()[2]-1)/7)
            len1 = int((coords1.size()[2]-1)/7)
            l0 = []
            l1 = []
            for i in self.fre1:
                l0.append(len0*i)
            for i in self.fre2:
                l1.append(len1*i)
            res0 = sel_freq(desc0, u0_0, l0)
            res1 = sel_freq(desc1, u0_1, l1)

        src_vectors = get_vector_and_coord(coords1)
        tgt_vectors = get_vector_and_coord(coords0)
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':

                desc0 = desc0 + layer(desc0, desc1)
                desc1 = desc1 + layer(desc1, desc0)
            elif name == 'self':
                desc0 = layer(coords0, desc0, src_embeddings, src_patterns, tgt_center2point, tgt_index, is_em=1)

                desc1 = layer(coords1, desc1, tgt_embeddings, src_patterns, tgt_center2point, tgt_index, max_pool_x, src_vectors, tgt_vectors, is_src=0, is_em=1)



            if(self.mode == 'test'):
                desc0 = self.channelattention(res0, desc0)
                desc1 = self.channelattention(res1, desc1)

        return desc0, desc1