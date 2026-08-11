from numpy.lib.function_base import append
import open3d as o3d
import numpy as np
from functools import partial
import torch
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from lib.timer import Timer
from lib.utils import load_obj, natural_key
from datasets.indoor import IndoorDataset
from datasets.kitti import KITTIDataset
from datasets.modelnet import get_train_datasets, get_test_datasets
import math as m
import os

def pointIsInsideCuboid(vertex, p):
    #首先判断点是否在左右两面的中间 此时法线为y轴
	#XYZ vector_DP
	#XYZ vector_EP
	#XYZ vector_DE #法线y
	#计算向量DE
    #vector_DE.x=cuboid[7].x-cuboid[4].x;vector_DE.y=cuboid[7].y-cuboid[4].y;vector_DE.z=cuboid[7].z-cuboid[4].z;

    vector_DE = vertex[1] - vertex[2]

	#计算向量DP
	#vector_DP.x=p.x-cuboid[3].x;vector_DP.y=p.y-cuboid[3].y;vector_DP.z=p.z-cuboid[3].z
    vector_DP = p-vertex[2]
	#计算向量EP
	#vector_EP.x=p.x-cuboid[2].x;vector_EP.y=p.y-cuboid[2].y;vector_EP.z=p.z-cuboid[2].z
    vector_EP = p-vertex[1]

	#计算向量点乘的结果
	#DP点乘DE
	#double DP_DE
    dp_de = vector_DP[0]*vector_DE[0]+vector_DP[1]*vector_DE[1]+vector_DP[2]*vector_DE[2]
	#EP点乘DE
	#double EP_DE
    ep_de = vector_EP[0]*vector_DE[0]+vector_EP[1]*vector_DE[1]+vector_EP[2]*vector_DE[2]

	#然后判断点是否在上下两面的中间 此时法线为z轴
	#XYZ vector_DP;   DP已经存在了 直接用上面的
	#XYZ vector_AP
	#XYZ vector_AD  #法线y
	#计算向量AP
	#vector_AP.x=p.x-cuboid[0].x;vector_AP.y=p.y-cuboid[0].y;vector_AP.z=p.z-cuboid[0].z
    vector_AP = p-vertex[6]
	#计算向量AD
	#vector_AD.x=cuboid[4].x-cuboid[0].x;vector_AD.y=cuboid[4].y-cuboid[0].y;vector_AD.z=cuboid[4].z-cuboid[0].z
    vector_AD = vertex[2]-vertex[6]
	#计算向量点乘的结果
	#AD AP
	#double AD_AP
    ad_ap=vector_AD[0]*vector_AP[0]+vector_AD[1]*vector_AP[1]+ vector_AD[2]*vector_AP[2]
	#AD DP
    #double AD_DP
    ad_dp=vector_AD[0]*vector_DP[0]+vector_AD[1]*vector_DP[1]+ vector_AD[2]*vector_DP[2]

	#最后判断点是否在前后两面的中间 此时法线为x轴 
	#XYZ vector_OA #法线
	#XYZ vector_OP
	#XYZ vector_AP   已有
	#vector_OA.x=cuboid[0].x;vector_OA.y=cuboid[0].y;vector_OA.z=cuboid[0].z
	#vector_OP.x=p.x;vector_OP.y=p.y;vector_OP.z=p.z
    vector_OA = vertex[6]-vertex[7]
    vector_OP = p-vertex[7]

	#计算向量点乘的结果
	#OP OA
	#double OP_OA
    op_oa=vector_OP[0]*vector_OA[0]+vector_OP[1]*vector_OA[1]+vector_OP[2]*vector_OA[2]
	#AP OA 
	#double AP_OA
    ap_oa=vector_AP[0]*vector_OA[0]+vector_AP[1]*vector_OA[1]+vector_AP[2]*vector_OA[2]
    th = 1e-6
    #print(dp_de*ep_de, ad_ap*ad_dp, op_oa*ap_oa)
    if (dp_de*ep_de<0 or abs(dp_de*ep_de) < th) and (ad_ap*ad_dp<0 or abs(ad_ap*ad_dp) < th) and (op_oa*ap_oa<0 or abs(op_oa*ap_oa)<th):
        return True
    else:
        return False
    
def distance(p1, p2):
    return m.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) + pow(p1[2] - p2[2], 2))

def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)

def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)
    
def collate_fn_descriptor(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []
    batched_src_loc_list = []
    batched_tgt_loc_list = []
    batched_src_all_loc_list = []
    batched_tgt_all_loc_list = []
    assert len(list_data) == 1
    # TODO: 会返回元数据
    for ind, (src_pcd,tgt_pcd,src_feats,tgt_feats,rot,trans,matching_inds, src_pcd_raw, tgt_pcd_raw, sample, base, src_name, tgt_name, src_loc, tgt_loc, src_all_loc, tgt_all_loc) in enumerate(list_data):
        batched_points_list.append(src_pcd)
        batched_points_list.append(tgt_pcd)
        batched_features_list.append(src_feats)
        batched_features_list.append(tgt_feats)
        batched_lengths_list.append(len(src_pcd))
        batched_lengths_list.append(len(tgt_pcd))
        if type(src_loc) == np.ndarray:
            batched_src_loc_list.append(src_loc)
            batched_tgt_loc_list.append(tgt_loc)
        if type(src_all_loc) == np.ndarray:
            batched_src_all_loc_list.append(src_all_loc)
            batched_tgt_all_loc_list.append(tgt_all_loc)
    
    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0))
    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r, neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius / 1.75
            #dl = 2 * r_normal / config.conv_radius / 1.5
            #dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)
            """ a = np.asarray(batched_points)
            b = np.asarray(pool_p)
            np.save("a", a)
            np.save("b", b) """

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r, neighborhood_limits[layer])
            
            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r, neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

    #tgt_point2center = {}
    """ tgt_center2point = {}
    tgt_points = input_points[3][int(input_batches_len[3][0]):] """

    """ for p in tgt_points:
        for info in tgt_vertex_infos:
            if pointIsInsideCuboid(info[0:8], np.asarray(p)):
                #tgt_point2center.append(np.append(np.asarray(p), info[8]))
                tgt_center2point[tuple(info[8][0], info[8][1], info[8][2])]
                tgt_center2point[] """

    """ for info in tgt_vertex_infos:
        #temp = info[8].reshape(1,3)
        points = []
        for p in tgt_points:
            if pointIsInsideCuboid(info[0:8], np.asarray(p)):
                points.append(tuple(np.asarray(p)))
                #temp = np.concatenate((temp, np.asarray(p).reshape(1,3)))
                tgt_point2center[tuple(np.asarray(p))] = tuple(info[8])
        tgt_center2point[tuple(info[8])] = tuple(points) """

    """ l1 = [i for i in range(len(tgt_points))]
    for info in tgt_vertex_infos:
        index = []
        for i in range(len(tgt_points)):
            if pointIsInsideCuboid(info[0:8], np.asarray(tgt_points[i])):
                if l1.count(i) == 1:
                    l1.remove(i)
                    index.append(i)
                
        tgt_center2point[tuple(info[8])] = tuple(index) """
    """ if len(l1) != len(tgt_points):
        np.save("more", np.asarray(tgt_points))
        np.save("less", np.asarray(tgt_points[l1])) """
    #assert(len(l1) == len(tgt_points))
    """ if len(l1) != 0:
        dis = 100.0
        for id in l1:
            for infos in tgt_vertex_infos:
                if distance(tgt_points[id], infos[8]) < dis:
                    dis = distance(tgt_points[id], infos[8])
                    center = infos[8]
            l = list(tgt_center2point[tuple(center)])
            l.append(id)
            tgt_center2point[tuple(center)] = tuple(l)
    l2 = []
    for key, point_index in tgt_center2point.items():
        l2.extend(list(point_index))
    index = [0 for x in range(len(l2))]
    for i in range(len(l2)):
        index[l2[i]] = i
    
    tgt_index = torch.from_numpy(np.asarray(index)) """

    
    ###############
    # Return inputs
    ###############TODO: 返回的元数据要从这里返回
    if len(batched_src_loc_list) != 0 and len(batched_src_all_loc_list) != 0:
        dict_inputs = {
            'points': input_points,
            'neighbors': input_neighbors,
            'pools': input_pools,
            'upsamples': input_upsamples,
            'features': batched_features.float(),
            'stack_lengths': input_batches_len,
            'rot': torch.from_numpy(rot),
            'trans': torch.from_numpy(trans),
            'correspondences': matching_inds,
            'src_pcd_raw': torch.from_numpy(src_pcd_raw).float(),
            'tgt_pcd_raw': torch.from_numpy(tgt_pcd_raw).float(),
            'sample': sample,
            'base': base,
            'src_name': src_name,
            'tgt_name': tgt_name,
            'src_loc': torch.from_numpy(np.asarray(batched_src_loc_list)).float(),
            'tgt_loc': torch.from_numpy(np.asarray(batched_tgt_loc_list)).float(),
            'src_all_loc': torch.from_numpy(np.asarray(batched_src_all_loc_list)).float(),
            'tgt_all_loc': torch.from_numpy(np.asarray(batched_tgt_all_loc_list)).float()
            #'src_patterns': src_patterns,
            #'tgt_point2center':tgt_point2center,
            #'tgt_center2point':tgt_center2point,
            #'tgt_index':tgt_index
            #'tgt_index':[]
        }
    else:
        dict_inputs = {
            'points': input_points,
            'neighbors': input_neighbors,
            'pools': input_pools,
            'upsamples': input_upsamples,
            'features': batched_features.float(),
            'stack_lengths': input_batches_len,
            'rot': torch.from_numpy(rot),
            'trans': torch.from_numpy(trans),
            'correspondences': matching_inds,
            'src_pcd_raw': torch.from_numpy(src_pcd_raw).float(),
            'tgt_pcd_raw': torch.from_numpy(tgt_pcd_raw).float(),
            'sample': sample,
            'base': base,
            'src_name': src_name,
            'tgt_name': tgt_name
            #'src_patterns': src_patterns,
            #'tgt_point2center':tgt_point2center,
            #'tgt_center2point':tgt_center2point,
            #'tgt_index':tgt_index
            #'tgt_index':[]
        }

    return dict_inputs

def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):
    timer = Timer()
    last_display = timer.total_time

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        timer.tic()
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)
        timer.toc()

        if timer.total_time - last_display > 0.1:
            last_display = timer.total_time
            print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits

def get_datasets(config):
    if(config.dataset=='indoor'):
        info_train = load_obj(config.train_info)
        info_val = load_obj(config.val_info)
        #info_benchmark = load_obj(f'configs/indoor/{config.benchmark}.pkl')
        # info_benchmark = load_obj(f'node_configs/test_info.pkl')
        if os.path.exists('node_configs/test_info.pkl'):
            info_benchmark = load_obj('node_configs/test_info.pkl')
        else:
            info_benchmark = load_obj(f'configs/indoor/3DMatch.pkl')

        train_set = IndoorDataset(info_train,config,data_augmentation=False)
        val_set = IndoorDataset(info_val,config,data_augmentation=False)
        benchmark_set = IndoorDataset(info_benchmark,config, data_augmentation=False)
    elif(config.dataset == 'kitti'):
        train_set = KITTIDataset(config,'train',data_augmentation=True)
        val_set = KITTIDataset(config,'val',data_augmentation=False)
        benchmark_set = KITTIDataset(config, 'test',data_augmentation=False)
    elif(config.dataset=='modelnet'):
        train_set, val_set = get_train_datasets(config)
        benchmark_set = get_test_datasets(config)
    else:
        raise NotImplementedError

    return train_set, val_set, benchmark_set



def get_dataloader(dataset, batch_size=1, num_workers=4, shuffle=True, neighborhood_limits=None):
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, dataset.config, collate_fn=collate_fn_descriptor)
    print("neighborhood:", neighborhood_limits)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/4
        collate_fn=partial(collate_fn_descriptor, config=dataset.config, neighborhood_limits=neighborhood_limits),
        drop_last=False
    )
    return dataloader, neighborhood_limits


if __name__ == '__main__':
    pass
