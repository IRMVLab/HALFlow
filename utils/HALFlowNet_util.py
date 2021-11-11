""" PointNet++ Layers
Original Author: Charles R. Qi
Modified by Xinrui Wu
Date: July 2020
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))

from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
import pointconv_util


def warping_layers( xyz1, upsampled_flow):

    return xyz1+upsampled_flow

def square_distance(src, dst):

    B = src.get_shape()[0].value
    N = src.get_shape()[1].value
    M = dst.get_shape()[1].value

    for i in range(B):
        ddd = dst[i, :, :]
        sss = src[i, :, :]
        dist_i = -2 * tf.matmul(sss, tf.transpose(ddd, [1, 0]))
        dist_i = tf.expand_dims(dist_i, axis = 0)
        if i == 0:
            dist = dist_i
        else:
            dist = tf.concat([dist, dist_i], axis = 0 )

    dist = dist + tf.reshape(tf.reduce_sum(src ** 2, axis = -1), [B, N, 1])
    dist = dist + tf.reshape(tf.reduce_sum(dst ** 2, axis = -1), [B, 1, M])
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    group_dist, group_idx = tf.nn.top_k(0-sqrdists, nsample)
    return (0-group_dist), group_idx



def cost_volume(warped_xyz, warped_points, f2_xyz, f2_points, nsample, nsample_q, mlp1, mlp2, is_training, bn_decay, scope, bn=True, pooling='max', knn=True, corr_func='elementwise_product' ):
    
    with tf.variable_scope(scope) as sc:   

        _, idx = knn_point(nsample, warped_xyz, warped_xyz)

        pc_xyz_grouped = group_point(warped_xyz, idx) #b n m 3  
        pc_points_grouped = group_point(warped_points, idx)
        
        pc_xyz_new = tf.tile( tf.expand_dims (warped_xyz, axis = 2), [1,1,nsample,1] )
        pc_points_new = tf.tile( tf.expand_dims (warped_points, axis = 2), [1,1,nsample,1] )

        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new####b , n ,m ,3

        pc_euc_diff = tf.sqrt(tf.reduce_sum(tf.square(pc_xyz_diff), axis=3, keep_dims=True) + 1e-20)

        pc_xyz_diff_concat = tf.concat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff], axis=3)

        b = pc_xyz_grouped.get_shape()[0].value
        n = pc_xyz_grouped.get_shape()[1].value
        m = nsample
        c = pc_points_grouped.get_shape()[3].value

        pi_xyz_grouped = tf.reshape(pc_xyz_grouped,[b,n*m,3])
        pi_points_grouped = tf.reshape(pc_points_grouped,[b,n*m,c])

        _, idx_q = knn_point(nsample_q, f2_xyz, pi_xyz_grouped)
        qi_xyz_grouped = group_point(f2_xyz, idx_q)
        qi_points_grouped = group_point(f2_points, idx_q)

        pi_xyz_expanded = tf.tile(tf.expand_dims(pi_xyz_grouped, 2), [1,1,nsample_q,1]) # batch_size, npoint*m, nsample, 3
        pi_points_expanded = tf.tile(tf.expand_dims(pi_points_grouped, 2), [1,1,nsample_q,1]) # batch_size, npoint*m, nsample, 3
        
        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded
        pi_euc_diff = tf.sqrt(tf.reduce_sum(tf.square(pi_xyz_diff), axis=[-1] , keep_dims=True) + 1e-20 )
        pi_xyz_diff_concat = tf.concat([pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff], axis=3)
        
        pi_feat_diff = tf.concat(axis=-1, values=[pi_points_expanded, qi_points_grouped])
        pi_feat1_new = tf.concat([pi_xyz_diff_concat, pi_feat_diff], axis=3) # batch_size, npoint*m, nsample, [channel or 1] + 3

        for j, num_out_channel in enumerate(mlp1):
            pi_feat1_new = tf_util.conv2d(pi_feat1_new, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='CV_%d'%(j), bn_decay=bn_decay)


        pi_xyz_encoding = tf_util.conv2d(pi_xyz_diff_concat, mlp1[-1], [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='CV_xyz', bn_decay=bn_decay)

        pi_concat = tf.concat([pi_xyz_encoding, pi_feat1_new], axis = 3)

        for j, num_out_channel in enumerate(mlp2):
            pi_concat = tf_util.conv2d(pi_concat, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='sum_CV_%d'%(j), bn_decay=bn_decay)
        WQ = tf.nn.softmax(pi_concat,dim=2)
            
        pi_feat1_new = WQ * pi_feat1_new
        pi_feat1_new = tf.reduce_sum(pi_feat1_new, axis=[2], keep_dims=False, name='avgpool_diff')#b, n*m, mlp1[-1]
        pi_feat1_new = tf.reshape(pi_feat1_new,[b,n,m,-1])


        pc_xyz_encoding = tf_util.conv2d(pc_xyz_diff_concat, mlp1[-1], [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='sum_xyz_encoding', bn_decay=bn_decay)


        pc_concat = tf.concat([pc_xyz_encoding, pc_points_new, pi_feat1_new], axis = -1)

        for j, num_out_channel in enumerate(mlp2):
            pc_concat = tf_util.conv2d(pc_concat, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=True, is_training=is_training,
                                        scope='sum_cost_volume_%d'%(j), bn_decay=bn_decay)
        WP = tf.nn.softmax(pc_concat,dim=2)
        pc_feat1_new = WP * pi_feat1_new
        pc_feat1_new = tf.reduce_sum(pc_feat1_new, axis=[2], keep_dims=False, name='sumpool_diff')#b*n*mlp2[-1]

    return pc_feat1_new   



def flow_predictor(flow_coarse, points_f1, upsampled_feat, cost_volume, mlp, is_training, bn_decay, scope, bn=True ):

    with tf.variable_scope(scope) as sc:

        points_concat = tf.concat(axis=-1, values=[flow_coarse, points_f1, cost_volume, upsampled_feat]) # B,ndataset1,nchannel1+nchannel2
        
        points_concat = tf.expand_dims(points_concat, 2)

        for i, num_out_channel in enumerate(mlp):
            points_concat = tf_util.conv2d(points_concat, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_predictor%d'%(i), bn_decay=bn_decay)
        points_concat = tf.squeeze(points_concat,[2])
      
    return points_concat


def sample_and_group(npoint, nsample, xyz, xyz_raw, label, points, knn=True, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor  channel——是否涉及local point features
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    
    if xyz_raw != None:
        new_xyz, _, _, _ = tf.split(xyz, num_or_size_splits = 4 , axis= 1)
        new_xyz_raw, _, _, _ = tf.split(xyz_raw, num_or_size_splits = 4 , axis= 1)
        new_label, _, _, _ = tf.split(label, num_or_size_splits = 4 , axis= 1)

    else:
        sample_idx = farthest_point_sample(npoint, xyz)
        new_xyz = gather_point(xyz, sample_idx) # (batch_size, npoint, 3)
        new_label = gather_point(label, sample_idx)

    # _, idx_q = knn_point(nsample, xyz, new_xyz)
    # xyz_grouped = group_point(xyz, idx_q)
    # grouped_points = group_point(points, idx_q)
    
    xyz_grouped, _, grouped_points, idx = pointconv_util.grouping(points, nsample, xyz, new_xyz)## b, npoints, n_sample_q c

    xyz_expanded = tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # batch_size, npoint*m, nsample, 3
    xyz_diff = xyz_grouped - xyz_expanded

    if points is None:
        new_points = tf.concat([xyz_diff, xyz_expanded] , axis=-1)
    
    else:
        new_points = tf.concat([xyz_diff, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)

    if xyz_raw != None:
        return new_xyz, new_label, new_points, new_xyz_raw
    else:
        return new_xyz, new_label, new_points


def pointnet_sa_module(xyz, xyz_raw, label, points, npoint, nsample, mlp, mlp2, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
            
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        if xyz_raw != None:
            new_xyz, new_label, new_points, new_xyz_raw = sample_and_group(npoint,  nsample, xyz, xyz_raw, label, points, knn, use_xyz)
        else:
            new_xyz, new_label, new_points = sample_and_group(npoint,  nsample, xyz, xyz_raw, label, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2]) 

        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=bn, is_training=is_training,
                                    scope='conv%d'%(i), bn_decay=bn_decay,
                                    data_format=data_format)

        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                            data_format=data_format)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])

        if xyz_raw != None:
            return new_xyz, new_label,  new_points, new_xyz_raw
        else:
            return new_xyz, new_label,  new_points 


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True, last_mlp_activation=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm 
        interpolated_points = three_interpolate(points2, idx, weight)

        new_points1 = interpolated_points

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            if i == len(mlp)-1 and not(last_mlp_activation):
                activation_fn = None #
            else:
                activation_fn = tf.nn.relu
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay, activation_fn=activation_fn)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1


def set_upconv_module(xyz1, xyz2, feat1, feat2, nsample, mlp, mlp2, is_training, scope, bn_decay=None, bn=True, pooling='max', knn=True):

    """
        Feature propagation from xyz2 (less points) to xyz1 (more points)

    Inputs:
        xyz1: (batch_size, npoint1, 3)
        xyz2: (batch_size, npoint2, 3)
        feat1: (batch_size, npoint1, channel1) features for xyz1 points (earlier layers)
        feat2: (batch_size, npoint2, channel2) features for xyz2 points
    Output:
        feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)

    """

    with tf.variable_scope(scope) as sc:

        _, idx_q = knn_point(nsample, xyz2, xyz1)
        xyz_grouped = group_point(xyz2, idx_q)
        feat2_grouped = group_point(feat2, idx_q)
        
        xyz_expanded = tf.tile(tf.expand_dims(xyz1, 2), [1,1,nsample,1]) # batch_size, npoint*m, nsample, 3
        xyz_diff = xyz_grouped - xyz_expanded
    
        net = tf.concat([feat2_grouped, xyz_diff], axis=3) # batch_size, npoint1, nsample, channel2+3

        if mlp is None: mlp=[]
        for i, num_out_channel in enumerate(mlp):
            net = tf_util.conv2d(net, num_out_channel, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv%d'%(i), bn_decay=bn_decay)
        if pooling=='max':
            feat1_new = tf.reduce_max(net, axis=[2], keep_dims=False, name='maxpool') # batch_size, npoint1, mlp[-1]
        elif pooling=='avg':
            feat1_new = tf.reduce_mean(net, axis=[2], keep_dims=False, name='avgpool') # batch_size, npoint1, mlp[-1]

        if feat1 is not None:
            feat1_new = tf.concat([feat1_new, feat1], axis=2) # batch_size, npoint1, mlp[-1]+channel1

        feat1_new = tf.expand_dims(feat1_new, 2) # batch_size, npoint1, 1, mlp[-1]+channel2
        if mlp2 is None: mlp2=[]
        for i, num_out_channel in enumerate(mlp2):
            feat1_new = tf_util.conv2d(feat1_new, num_out_channel, [1,1],
                                       padding='VALID', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='post-conv%d'%(i), bn_decay=bn_decay)
        feat1_new = tf.squeeze(feat1_new, [2]) # batch_size, npoint1, mlp2[-1]

        return feat1_new

