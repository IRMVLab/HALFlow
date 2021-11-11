"""
    FlowNet3D model with up convolution
"""

import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
from HALFlowNet_util import *

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * 2, 6))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_pl_raw = tf.placeholder(tf.float32, shape=(batch_size, num_point * 2, 6))
    return pointclouds_pl, labels_pl, pointclouds_pl_raw


def get_model(point_cloud, point_cloud_raw, label, is_training, dataset = 'ft3d', bn_decay=None):

    if dataset == 'kitti':
        l0_npoint = 4096; nsample_list = [64, 64] 
    
    elif dataset == 'ft3d':
        l0_npoint = 2048; nsample_list = [32, 24]

    num_point = point_cloud.get_shape()[1].value // 2

    l0_xyz_f1 = point_cloud[:, :num_point, 0:3]
    l0_xyz_f1_raw = point_cloud_raw[:, :num_point, 0:3]

    l0_points_f1 = point_cloud[:, :num_point, 3:]

    l0_label_f1 = label

    l0_xyz_f2 = point_cloud[:, num_point:, 0:3]
    l0_xyz_f2_raw = point_cloud_raw[:, num_point:, 0:3]

    l0_points_f2 = point_cloud[:, num_point:, 3:]

    with tf.variable_scope('sa1') as scope:

        l0_xyz_f1, l0_label_f1, l0_points_f1, pc1_sample = pointnet_sa_module( l0_xyz_f1, l0_xyz_f1_raw, l0_label_f1, l0_points_f1, npoint=l0_npoint, nsample=nsample_list[0], mlp=[16,16,32], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer0')
        l1_xyz_f1, l1_label, l1_points_f1 = pointnet_sa_module( l0_xyz_f1, None, l0_label_f1, l0_points_f1, npoint=1024,  nsample=nsample_list[1], mlp=[32,32,64], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        l2_xyz_f1, l2_label, l2_points_f1= pointnet_sa_module( l1_xyz_f1, None, l1_label, l1_points_f1, npoint=256, nsample=16, mlp=[64,64,128], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l3_xyz_f1, l3_label, l3_points_f1= pointnet_sa_module( l2_xyz_f1, None, l2_label, l2_points_f1, npoint=64, nsample=16, mlp=[128,128,256], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer3')

        scope.reuse_variables()
        
        l0_xyz_f2,_, l0_points_f2, pc2_sample = pointnet_sa_module( l0_xyz_f2, l0_xyz_f2_raw, label, l0_points_f2, npoint=l0_npoint,  nsample=nsample_list[0], mlp=[16,16,32],   mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer0')
        l1_xyz_f2,_, l1_points_f2= pointnet_sa_module( l0_xyz_f2, None, l0_label_f1, l0_points_f2, npoint=1024,  nsample=nsample_list[1], mlp=[32,32,64],   mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer1')
        l2_xyz_f2,_, l2_points_f2= pointnet_sa_module( l1_xyz_f2, None, l1_label, l1_points_f2, npoint=256, nsample=16, mlp=[64,64,128], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l3_xyz_f2,_, l3_points_f2= pointnet_sa_module( l2_xyz_f2, None, l2_label, l2_points_f2, npoint=64, nsample=16, mlp=[128,128,256], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    l2_points_f1_new = cost_volume( l2_xyz_f1, l2_points_f1, l2_xyz_f2, l2_points_f2, nsample=4, nsample_q=32, mlp1=[256,128,128], mlp2 = [256,128], is_training=is_training, bn_decay=bn_decay, scope='flow_embedding_2', bn=True, pooling='max', knn=True, corr_func='concat')
    # Layer 3
    l3_xyz_f1,l3_label, l3_points_f1 = pointnet_sa_module( l2_xyz_f1, None, l2_label, l2_points_f1_new, npoint=64, nsample=16, mlp=[128,128,256], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer3_flow')
    
    # Layer 4
    l4_xyz_f1, _, l4_points_f1 = pointnet_sa_module( l3_xyz_f1, None, l3_label, l3_points_f1, npoint=16, nsample=8, mlp=[256,256,512], mlp2=None, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation
    l3_points_f1_new = set_upconv_module( l3_xyz_f1, l4_xyz_f1, l3_points_f1, l4_points_f1, nsample=8, mlp=[256,256,512], mlp2=[512], scope='up_sa_layer1', is_training=is_training, bn_decay=bn_decay, knn=True)
    

    #####layer3
    
    l3_flow_coarse = tf_util.conv1d(l3_points_f1_new, 3, 1, padding='VALID', activation_fn=None, scope='fc2_flow3')###activation_fn = None
    
    l3_flow_warped = warping_layers(l3_xyz_f1, l3_flow_coarse)
    
    l3_cost_volume = cost_volume( l3_flow_warped, l3_points_f1, l3_xyz_f2, l3_points_f2, nsample=4, nsample_q=6, mlp1=[512,256,256], mlp2=[512,256], is_training=is_training, bn_decay=bn_decay, scope='l3_cost_volume', bn=True, pooling='max', knn=True, corr_func='concat')#b*n*mlp[-1
    
    l3_flow_finer = flow_predictor(l3_flow_coarse, l3_points_f1, l3_points_f1_new, l3_cost_volume, mlp=[512,256,256], is_training = is_training , bn_decay = bn_decay, scope='l3_finer')

    l3_flow = tf_util.conv1d(l3_flow_finer, 3, 1, padding='VALID', activation_fn=None, scope='flow3')###activation_fn = None


    #####layer 2

    l2_points_f1_new = set_upconv_module( l2_xyz_f1, l3_xyz_f1, l2_points_f1, l3_flow_finer, nsample=8, mlp=[256,128,128], mlp2=[128], scope='up_sa_layer2', is_training=is_training, bn_decay=bn_decay, knn=True)

    l2_flow_coarse = pointnet_fp_module(l2_xyz_f1, l3_xyz_f1, None, l3_flow, [], is_training, bn_decay, scope='coarse_l2')

    l2_flow_warped = warping_layers(l2_xyz_f1, l2_flow_coarse)

    l2_cost_volume = cost_volume( l2_flow_warped, l2_points_f1, l2_xyz_f2, l2_points_f2, nsample=4, nsample_q=6, mlp1=[256,128,128], mlp2=[256,128], is_training=is_training, bn_decay=bn_decay, scope='l2_cost_volume', bn=True, pooling='max', knn=True, corr_func='concat')#b*n*mlp[-1

    l2_flow_finer = flow_predictor(l2_flow_coarse, l2_points_f1, l2_points_f1_new, l2_cost_volume, mlp=[256,128,128], is_training = is_training , bn_decay = bn_decay, scope='l2_finer')

    l2_flow = tf_util.conv1d(l2_flow_finer, 3, 1, padding='VALID', activation_fn=None, scope='flow2')###activation_fn = None

    
    #####layer 1
    l1_points_f1_new = set_upconv_module( l1_xyz_f1, l2_xyz_f1, l1_points_f1, l2_flow_finer, nsample=8, mlp=[256,128,128], mlp2=[128], scope='up_sa_layer1_folw', is_training=is_training, bn_decay=bn_decay, knn=True)

    l1_flow_coarse = pointnet_fp_module(l1_xyz_f1, l2_xyz_f1, None, l2_flow, [], is_training, bn_decay, scope='coarse_l1')

    l1_flow_warped = warping_layers(l1_xyz_f1, l1_flow_coarse)

    l1_cost_volume = cost_volume( l1_flow_warped, l1_points_f1, l1_xyz_f2, l1_points_f2, nsample=4, nsample_q=6, mlp1=[128,64,64], mlp2=[128,64], is_training=is_training, bn_decay=bn_decay, scope='l1_cost_volume', bn=True, pooling='max', knn=True, corr_func='concat')#b*n*mlp[-1

    l1_flow_finer = flow_predictor(l1_flow_coarse, l1_points_f1, l1_points_f1_new, l1_cost_volume, mlp=[256,128,128], is_training = is_training , bn_decay = bn_decay, scope='l1_finer')

    l1_flow = tf_util.conv1d(l1_flow_finer, 3, 1, padding='VALID', activation_fn=None, scope='flow1')###activation_fn = None

    
    #####layer 0
    l0_feat_f1 = pointnet_fp_module(l0_xyz_f1, l1_xyz_f1, l0_points_f1, l1_flow_finer, [256,256], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_feat_f1, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    l0_flow = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc2')

    return l0_flow, l1_flow, l2_flow, l3_flow, l0_label_f1, l1_label, l2_label, l3_label, pc1_sample, pc2_sample



def get_loss(l0_pred, l1_pred, l2_pred, l3_pred, l0_label, l1_label, l2_label, l3_label):#####idx来选择真值

    l0_loss = tf.reduce_mean( tf.reduce_sum((l0_pred-l0_label) * (l0_pred-l0_label), axis=2) / 2.0)
    tf.summary.scalar('l0 loss', l0_loss)

    l1_loss = tf.reduce_mean( tf.reduce_sum((l1_pred-l1_label) * (l1_pred-l1_label), axis=2) / 2.0)
    tf.summary.scalar('l1 loss', l0_loss)

    l2_loss = tf.reduce_mean( tf.reduce_sum((l2_pred-l2_label) * (l2_pred-l2_label), axis=2) / 2.0)
    tf.summary.scalar('l2 loss', l2_loss)

    l3_loss = tf.reduce_mean( tf.reduce_sum((l3_pred-l3_label) * (l3_pred-l3_label), axis=2) / 2.0)
    tf.summary.scalar('l3 loss', l3_loss)

    loss_sum = 1.6*l3_loss + 0.8*l2_loss + 0.4*l1_loss + 0.2*l0_loss 

    tf.add_to_collection('losses', loss_sum)
    return loss_sum

