import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module


def placeholder_inputs(batch_size, num_point, num_features):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_features))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model_origin(point_cloud, is_training, num_features, num_classes, bn_decay=None):
    """ 
    Part segmentation based on PointNet++ (three S&G and feature propagation layers)
     - Input: B*N*NUM_FEATURES
     - Output: B*NUM_CLASSES 
    """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1, -1, num_features-3])

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=1, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=2, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    
    # Feature Propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')
    
    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_classes, 1, padding='VALID', activation_fn=None, scope='fc2')
    
    return net, end_points


def get_model_simplify(point_cloud, is_training, num_features, num_classes, bn_decay=None):
    """ 
    Part segmentation based on PointNet++ (two S&G and feature propagation layers)
     - Input: B*N*NUM_FEATURES
     - Output: B*NUM_CLASSES 
    """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1, -1, num_features-3])

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=1, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    
    # Feature Propagation layers
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer1')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer2')
    
    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_classes, 1, padding='VALID', activation_fn=None, scope='fc2')
    
    return net, end_points


def get_loss(pred, label, num_classes):
    """ 
        pred: B*N*NUM_CLASSES,
        label: B*N
    """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    
    # - Debug Port -
    debug_port = tf.constant(1)
    # --------------
    
    return classify_loss, debug_port


def get_focal_loss_v1(pred, label, gamma, alpha, num_classes):
    """ 
    Class-balance weight - Type 1:
        pred: B*N*NUM_CLASSES,
        label: B*N
        alpha: a NUM_CLASSES dimension vector indicating the weights of individual classes
    """
    
    # - Parameter setting for focal loss
    epsilon = 1.e-7
    gamma = tf.constant(gamma)
    weight_cls = tf.constant(alpha, shape=(1, num_classes, 1), dtype=tf.float32)
    weight_cls_tile = tf.tile(weight_cls, (tf.shape(label)[0], 1, 1))  # BxCx1
    
    # - Get cross entropy loss
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)  # BxN
    ce_loss_reshape = tf.reshape(ce_loss, [tf.shape(label)[0], 1, tf.shape(label)[1]])  # Bx1xN
    
    # - Focal loss: Focal weights with both class-balance weights and tough-instance weight
    pred_softmax = tf.nn.softmax(pred)
    pred_softmax_clip = tf.clip_by_value(pred_softmax, epsilon, 1. - epsilon)
    label_class_one_hot = tf.one_hot(label, num_classes)  # BxNxC, dtype=float32
    
    focal_weights_for_tough_instance = tf.multiply(label_class_one_hot, \
                                                   tf.pow(tf.subtract(1., pred_softmax_clip), gamma))  # BxNxC
    
    focal_weights = tf.matmul(focal_weights_for_tough_instance, weight_cls_tile)  # BxNx1
    
    focal_loss = tf.matmul(ce_loss_reshape, focal_weights)  # Bx1x1
    
    
    classify_loss = tf.reduce_mean(focal_loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    
    # - Debug Port -
    debug_port = tf.shape(1)
    # --------------
    
    return classify_loss, debug_port


def get_focal_loss_v2(pred, label, gamma, alpha, num_classes):
    """ 
    Class-balance weight - Type 2:
        pred: B*N*NUM_CLASSES,
        label: B*N
        alpha: a single float scalar value indicating how much you would like 
               the network to concentrate on the ground truth positive candidates.
    """
    
    # - Parameter setting for focal loss
    epsilon = 1.e-7
    gamma = tf.constant(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)
    
    # - Get cross entropy loss
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)  # BxN
    ce_loss_reshape = tf.reshape(ce_loss, [tf.shape(label)[0], tf.shape(label)[1], 1])  # BxNx1
    ce_loss_tile = tf.tile(ce_loss_reshape, (1, 1, num_classes))  # BxNxC
    
    # - Get focal loss: Focal weights with both class-balance weights and tough-instance weight
    pred_softmax = tf.nn.softmax(pred)  # BxNxC
    pred_softmax_clip = tf.clip_by_value(pred_softmax, epsilon, 1. - epsilon)
    
    label_class_one_hot = tf.one_hot(label, num_classes)  # BxNxC, dtype=float32
    
    focal_weights_for_tough_instance = tf.multiply(label_class_one_hot, \
                                                   tf.pow(tf.subtract(1., pred_softmax_clip), gamma))  # BxNxC
        
    alpha_t = label_class_one_hot * alpha + (tf.ones_like(label_class_one_hot) - label_class_one_hot) * (1 - alpha)  # BxNxC
    
    focal_weights = tf.multiply(focal_weights_for_tough_instance, alpha_t)  # BxNxC
    
    focal_loss = tf.multiply(ce_loss_tile, focal_weights)  # BxNxC
    
    
    classify_loss = tf.reduce_mean(focal_loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    
    # - Debug Port -
    debug_port = tf.shape(alpha_t)
    # --------------
    
    return classify_loss, debug_port


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((5, 50000, 51))
        net, _ = get_model_origin(inputs, tf.constant(True))
        print(net)
