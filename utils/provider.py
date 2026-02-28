import os
import open3d as o3d
import sys
import numpy as np
import random
import h5py
import copy
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:,idx,:]

def rotate_point_cloud_xy(batch_data, sigma=0.05):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    B, N, C = batch_data.shape
    rotation_angles = sigma * np.random.randn(B)
    # rotation_angles = np.clip(rotation_angles, -1*clip, clip)
    is_rotate_y_bools = np.random.randint(low=0,high=2,size=(1, B),dtype='int')
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = rotation_angles[k]
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        if is_rotate_y_bools[0, k] == 0:
            rotation_matrix = np.array([[1, 0, 0],
                                        [0, cosval, -sinval],
                                        [0, sinval, cosval]])
        else:
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2.0 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k,:,0:3]
        shape_normal = batch_xyz_normal[k,:,3:6]
        batch_xyz_normal[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal

def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1,3)), rotation_matrix)
    return rotated_data



def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def random_scale_point_cloud_xy(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud with respect to X and Y axis. Scale is 
        per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    for batch_index in range(B):
        scales = np.zeros((3, 3), dtype="float32")
        scales[0, 0] = np.random.uniform(scale_low, scale_high, 1)
        scales[1, 1] = np.random.uniform(scale_low, scale_high, 1)
        scales[2, 2] = 1
        batch_data[batch_index,:,:] = np.dot(batch_data[batch_index,:,:], scales)
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc


def cropping(pc, max_cropping_ratio_xy=0.05, max_cropping_ratio_z=0.05):
    
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc[:, 0:3])
    pc_o3d_axis_aligned_bounding_box = np.asarray(o3d.geometry.AxisAlignedBoundingBox.create_from_points(pc_o3d.points).get_box_points())
    x_range = np.max(pc_o3d_axis_aligned_bounding_box[:, 0]) - np.min(pc_o3d_axis_aligned_bounding_box[:, 0])
    y_range = np.max(pc_o3d_axis_aligned_bounding_box[:, 1]) - np.min(pc_o3d_axis_aligned_bounding_box[:, 1])
    z_range = np.max(pc_o3d_axis_aligned_bounding_box[:, 2]) - np.min(pc_o3d_axis_aligned_bounding_box[:, 2])
    
    cropping_ratio_xy = np.random.uniform(0, max_cropping_ratio_xy, 1)
    cropping_ratio_z = np.random.uniform(0, max_cropping_ratio_z, 1)
    
    cropping_x_lower = np.min(pc_o3d_axis_aligned_bounding_box[:, 0]) + x_range * cropping_ratio_xy
    cropping_x_upper = np.max(pc_o3d_axis_aligned_bounding_box[:, 0]) - x_range * cropping_ratio_xy
    cropping_y_lower = np.min(pc_o3d_axis_aligned_bounding_box[:, 1]) + y_range * cropping_ratio_xy
    cropping_y_upper = np.max(pc_o3d_axis_aligned_bounding_box[:, 1]) - y_range * cropping_ratio_xy
    cropping_z_upper = np.max(pc_o3d_axis_aligned_bounding_box[:, 2]) - z_range * cropping_ratio_z
    
    pc_cropping = pc[(pc[:, 0] > cropping_x_lower) & (pc[:, 0] < cropping_x_upper) & \
                     (pc[:, 1] > cropping_y_lower) & (pc[:, 1] < cropping_y_upper) & \
                     (pc[:, 2] < cropping_z_upper), :]
        
    return pc_cropping


def random_down_sampling(pc, num_point):
    """
    Randomly select num_point points from the original point cloud
    Input:
        pc: original point cloud with the size of N*num_features
        num_point: a scalar indicating the number of points to be selected
    Return:
        Down sampled point clouds with uniform point numbers, shape: num_point*num_features
    """
    selected_id = np.random.choice(range(pc.shape[0]), size=num_point, replace=None, p=None)
    pc_down_sample = pc[selected_id,]
    return pc_down_sample


def brightness_transform(pc_col, brightness_transform_range=[0.7, 2.0]):
    batch_size = pc_col.shape[0]
    brightness_aug_ratio = np.random.uniform(brightness_transform_range[0], brightness_transform_range[1], batch_size)
    for batch_index in range(batch_size):
        pc_col[batch_index, :, :] = np.clip(pc_col[batch_index, :, :] * brightness_aug_ratio[batch_index], 0, 1)
    return pc_col


def leaf_shift(pc_pos, instance_label, start_rank, n_leaves, sigma=0.05, clip=0.25):
    """
    Randomly shift the position of individual leaves within a certain range.
    Input:
        pc_pos: Nx3 array, original point positional info
        instance_label: labels of instances for individual points
        start_rank: rank of the first effective leaf
        n_leaves: total number of effective leaves within the plant point cloud
        sigma: standard deviation for the shifting Gaussian distribution
        clip: correct shifting values that is too large
    Return:
        Nx3 array, positional info of points after leaf translation operation
    """
    shifts_origin = sigma * np.random.randn(n_leaves)
    shifts = np.clip(shifts_origin, -1*clip, clip)
    for leaf_id in range(start_rank, start_rank + n_leaves):
        pc_pos[instance_label == leaf_id + 6, 2] += shifts[leaf_id - start_rank]
    return pc_pos


def leaf_rotate_z(pc_pos, instance_label, start_rank, leaf_base, n_leaves, rotate_range):
    """ 
    Randomly rotate the leaves within the plant along the vertical axis at the
    leaf base point to augument the dataset. The rotation angle is randomly
    selected with a uniform distribution.
    Input:
        pc_pos: Nx3 array, original point positional info
        instance_label: labels of instances for individual points
        start_rank: rank of the first effective leaf
        leaf_base: coordinate of the connection point between the leaf and the stem
        n_leaves: total number of effective leaves within the plant point cloud
        rotate_range: range of the uniform distribution for the rotation angle selection.
    Return:
        Nx3 array, positional info of points after leaf rotation operation
    """
    rotation_angles = np.random.uniform(-1, 1, n_leaves) * rotate_range
    for leaf_id in range(start_rank, start_rank + n_leaves):
        rotation_angle = rotation_angles[leaf_id - start_rank]
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        leaf_base_xy0 = np.tile(np.array([[leaf_base[leaf_id * 6], leaf_base[leaf_id * 6 + 1], 0]]), \
                                (pc_pos[instance_label == leaf_id + 6, :].shape[0], 1))
        pc_z_centered = pc_pos[instance_label == leaf_id + 6, 0:3] - leaf_base_xy0
        pc_pos[instance_label == leaf_id + 6, 0:3] = np.dot(pc_z_centered, rotation_matrix) + leaf_base_xy0
    return pc_pos


def leaf_rotate_pca(pc_pos, instance_label, start_rank, leaf_base, n_leaves, rotate_range):
    """ 
    Randomly rotate the leaves within the plant along the principle axis of the
    leaf instance to augument the dataset. The rotation angle is randomly
    selected with a uniform distribution.
    Input:
        pc_pos: Nx3 array, original point positional info
        instance_label: labels of instances for individual points
        start_rank: rank of the first effective leaf
        leaf_base: coordinate of the connection point between the leaf and the stem
        n_leaves: total number of effective leaves within the plant point cloud
        rotate_range: range of the uniform distribution for the rotation angle selection.
    Return:
        Nx3 array, positional info of points after leaf rotation operation
    """
    rotation_angles = np.random.uniform(-1, 1, n_leaves) * rotate_range
    
    for leaf_id in range(start_rank, start_rank + n_leaves):
        
        rotation_angle = rotation_angles[leaf_id - start_rank]
        
        if abs(rotation_angle) < 1e-9:
            rotation_angle = 1e-9 * (rotation_angle/abs(rotation_angle))
        
        leaf_orientation = np.array([leaf_base[leaf_id * 6 + 3], leaf_base[leaf_id * 6 + 4], leaf_base[leaf_id * 6 + 5]])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(leaf_orientation * rotation_angle)
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(pc_pos[instance_label == leaf_id + 6, 0:3])
        pc_o3d.rotate(rotation_matrix)
        pc_pos[instance_label == leaf_id + 6, 0:3] = np.array(pc_o3d.points)
        
    return pc_pos


def leaf_crossover(all_raw_batch_pcs, batch_pc_file_name, \
                   batch_cultivars, batch_leaf_numbers, batch_start_rank, batch_leaf_base, \
                   cultivar, num_features, n_max_crossover_leaves=4):
    
    # try:
    cultivar_start_rank = batch_start_rank[batch_cultivars == cultivar]
    cultivar_leaf_numbers = batch_leaf_numbers[batch_cultivars == cultivar]
    cultivar_end_rank = cultivar_leaf_numbers
    crossover_start_rank = cultivar_start_rank.max()
    crossover_end_rank = cultivar_end_rank.min()
    n_crossover_leaves = np.min([crossover_end_rank - crossover_start_rank + 1, n_max_crossover_leaves])
    rank_crossover = random.sample(list(range(crossover_start_rank, crossover_end_rank + 1)), n_crossover_leaves)  # Randomly select ranks to conduct crossover operation
    idx = np.where(batch_cultivars == cultivar)[0].tolist()
    for rank_id in rank_crossover:
        # Cut the leaves being selected from the original point cloud and 
        # collect them in the pool for concatenation.
        leaf_feature_pool = list()
        leaf_instance_label_pool = list()
        leaf_semantic_label_pool = list()
        leaf_pc_id = list()
        for pc_id in range(len(idx)):
            pc_this = copy.deepcopy(all_raw_batch_pcs[idx[pc_id]])
            start_rank_this = cultivar_start_rank.iloc[pc_id]
            # If there are missing leaves at the bottom of a plant, the 
            # leaf index to be cut off should be counted from the first 
            # recognised leaf. The start rank just refers to the index of 
            # the first leaf that is able to be recognised. Here, start rank
            # 1 refers to instance label 7.
            exclude_id = np.where(pc_this[:, -2] == rank_id + start_rank_this + 5)[0].tolist()
            leaf_feature_pool.append(pc_this[exclude_id, 0:num_features])
            leaf_instance_label_pool.append(pc_this[exclude_id, -2].reshape([-1, 1]))
            leaf_semantic_label_pool.append(pc_this[exclude_id, -1].reshape([-1, 1]))
            leaf_pc_id.append(pc_id)  # The original parent point cloud ID of the leaf being collected in the pool. This ID refers to the index of "idx", not the point clouds in the batch.
            all_raw_batch_pcs[idx[pc_id]] = np.delete(pc_this, exclude_id, axis=0)
        # Crossover operation
        for pc_id in range(len(idx)):
            pc_this = copy.deepcopy(all_raw_batch_pcs[idx[pc_id]])
            potential_pc_id_of_leaf = list(np.delete(np.array(leaf_pc_id), np.array(leaf_pc_id) == pc_id))  # The variable refers to the index of "idx", also refers to the index of leaves in the leaf pool.
            selected_pc_id_of_leaf = random.sample(potential_pc_id_of_leaf, 1)  # Select the leaf to be concatenated. This leaf id refers to the pc_id which the leaf comes from.
            potential_pc_id_of_leaf.remove(selected_pc_id_of_leaf)
            start_rank_this = cultivar_start_rank.iloc[pc_id]
            start_rank_concate = cultivar_start_rank.iloc[selected_pc_id_of_leaf[0]]
            leaf_base_this = batch_leaf_base.iloc[(rank_id + start_rank_this - 1) * 6 : \
                                                  (rank_id + start_rank_this - 1) * 6 + 3, idx[pc_id]].values  # Leaf base of on the plant which is going to be as the main plant during the concatenate operation.
            leaf_base_concate = batch_leaf_base.iloc[(rank_id + start_rank_concate - 1) * 6 : \
                                                      (rank_id + start_rank_concate - 1) * 6 + 3, idx[selected_pc_id_of_leaf[0]]].values  # Leaf base of the leaf which is going to be concatenated to the plant.
            orientation_vec_this = batch_leaf_base.iloc[(rank_id + start_rank_this - 1) * 6 + 3 : \
                                                        (rank_id + start_rank_this - 1) * 6 + 6, idx[pc_id]].values
            orientation_vec_concate = batch_leaf_base.iloc[(rank_id + start_rank_concate - 1) * 6 + 3 : \
                                                            (rank_id + start_rank_concate - 1) * 6 + 6, idx[selected_pc_id_of_leaf[0]]].values
            
            ## Debug ## -------------------------------------------------------
            leaf_base_concate_old = batch_leaf_base.iloc[rank_id * 6 : rank_id * 6 + 3, idx[selected_pc_id_of_leaf[0]]].values
            if sum(np.isnan(leaf_base_concate)) != 0:
                print("plant this: ", list(batch_start_rank.index)[idx[pc_id]])
                print("leaf source: ", list(batch_start_rank.index)[idx[selected_pc_id_of_leaf[0]]])
                print("cultivar_start_rank: ", cultivar_start_rank, "; cultivar_end_rank: ", cultivar_end_rank)
                print("crossover_start_rank: ", crossover_start_rank, "; crossover_end_rank: ", crossover_end_rank)
                print("rank_id", rank_id)
                print(leaf_base_concate, leaf_base_concate_old)
            # orientation_vec_concate_old = batch_leaf_base.iloc[rank_id * 6 + 3 : rank_id * 6 + 6, idx[selected_pc_id_of_leaf[0]]].values
            ## ----------------------------------------------------------------
            # Rotate the leaf being concatenated to make its main axis 
            # overlap with the original leaf.
            ori_vec_01 = np.zeros((1, 3), dtype="float32")
            ori_vec_02 = np.zeros((1, 3), dtype="float32")
            ori_vec_01[0, 0:2] = orientation_vec_this[0:2].reshape([1, -1])
            ori_vec_02[0, 0:2] = orientation_vec_concate[0:2].reshape([1, -1])
            cosval = np.dot(ori_vec_01[0, 0:2], ori_vec_02[0, 0:2])  # Both orientation vectors have a normal of 1.
            cross_product = np.cross(ori_vec_02, ori_vec_01)
            sinval = np.sqrt(1 - np.power(cosval, 2)) * (cross_product[0, -1]/abs(cross_product[0, -1]))  # The last term in the bracket is used to determine whether the value of sin() is positive (or negative).
            rotation_matrix = np.array([[cosval, sinval, 0],
                                        [-sinval, cosval, 0],
                                        [0, 0, 1]])
            leaf_feature_concate = copy.deepcopy(leaf_feature_pool[selected_pc_id_of_leaf[0]])
            concate_leaf_base_xy0 = np.tile(np.array([[leaf_base_concate[0], leaf_base_concate[1], 0]]), \
                                           (leaf_feature_concate.shape[0], 1))
            pc_z_centered = leaf_feature_concate[:, 0:3] - concate_leaf_base_xy0  # Move the leaf being concatenated to the main plant to the origin point of the axis, in order to conduct further rotation.
            shift_info = np.tile(np.array([[leaf_base_this[0], leaf_base_this[1], leaf_base_this[2] - leaf_base_concate[2]]]), \
                                (leaf_feature_concate.shape[0], 1))
            leaf_feature_concate[:, 0:3] = np.dot(pc_z_centered, rotation_matrix) + shift_info
            
            ## Debug ## -------------------------------------------------------
            id_nan = np.where(np.isnan(leaf_feature_concate))[0]
            if len(id_nan) != 0:
                print("rotation_matrix: ", rotation_matrix)
                print('######################################################')
            ## ----------------------------------------------------------------
            
            # Align instance labels for each crossovered leaf, the label is 
            # coincident with the removed leaf that was originally on that plant.
            new_instant_label = rank_id + start_rank_this + 5
            concate_leaf_instance_labels = new_instant_label * np.ones((leaf_feature_concate.shape[0], 1), dtype="float32")
            
            pc_leaf_crossovered = np.concatenate((leaf_feature_concate, \
                                                  concate_leaf_instance_labels, \
                                                  leaf_semantic_label_pool[selected_pc_id_of_leaf[0]]), axis=1)
            
            all_raw_batch_pcs[idx[pc_id]] = np.concatenate((pc_this, pc_leaf_crossovered), axis=0)
                
    # except:
    #     print("ERROR!!!")
    #     print(batch_pc_file_name)
    #     print("Cultivar: ", cultivar)
    #     print("Range of leaf ranks for crossover: ", [crossover_start_rank, crossover_end_rank])
    #     print("Selected crossover ranks: ", rank_crossover)
    #     print("start_rank_this", start_rank_this)
    #     print("start_rank_concate", start_rank_concate)
    #     os._exit()
    
    return all_raw_batch_pcs


def leaf_crossover_old(all_raw_batch_pcs, batch_pc_file_name, \
                   batch_cultivars, batch_leaf_numbers, batch_start_rank, batch_leaf_base, \
                   cultivar, num_features, n_max_crossover_leaves=4):
    
    try:
        cultivar_start_rank = batch_start_rank[batch_cultivars == cultivar]
        cultivar_leaf_numbers = batch_leaf_numbers[batch_cultivars == cultivar]
        cultivar_end_rank = cultivar_start_rank + cultivar_leaf_numbers - 1
        crossover_start_rank = cultivar_start_rank.max()
        crossover_end_rank = cultivar_end_rank.min()
        n_crossover_leaves = np.min([crossover_end_rank - crossover_start_rank + 1, n_max_crossover_leaves])
        rank_crossover = random.sample(list(range(crossover_start_rank, crossover_end_rank + 1)), n_crossover_leaves)  # Randomly select ranks to conduct crossover operation
        idx = np.where(batch_cultivars == cultivar)[0].tolist()
        for rank_id in rank_crossover:
            # Cut the leaves being selected from the original point cloud and 
            # collect them in the pool for concatenation.
            leaf_feature_pool = list()
            leaf_instance_label_pool = list()
            leaf_semantic_label_pool = list()
            leaf_pc_id = list()
            for pc_id in range(len(idx)):
                pc_this = copy.deepcopy(all_raw_batch_pcs[idx[pc_id]])
                start_rank_this = cultivar_start_rank.iloc[pc_id]
                exclude_id = np.where(pc_this[:, -2] == rank_id + start_rank_this + 5)[0].tolist()
                leaf_feature_pool.append(pc_this[exclude_id, 0:num_features])
                leaf_instance_label_pool.append(pc_this[exclude_id, -2].reshape([-1, 1]))
                leaf_semantic_label_pool.append(pc_this[exclude_id, -1].reshape([-1, 1]))
                leaf_pc_id.append(pc_id)  # The original parent point cloud ID of the leaf being collected in the pool. This ID refers to the index of "idx", not the point clouds in the batch.
                all_raw_batch_pcs[idx[pc_id]] = np.delete(pc_this, exclude_id, axis=0)
            # Crossover operation
            for pc_id in range(len(idx)):
                pc_this = copy.deepcopy(all_raw_batch_pcs[idx[pc_id]])
                potential_pc_id_of_leaf = list(np.delete(np.array(leaf_pc_id), np.array(leaf_pc_id) == pc_id))  # The variable refers to the index of "idx", also refers to the index of leaves in the leaf pool.
                selected_pc_id_of_leaf = random.sample(potential_pc_id_of_leaf, 1)
                leaf_base_this = batch_leaf_base.iloc[rank_id * 6 : rank_id * 6 + 3, idx[pc_id]].values  # Leaf base of on the plant which is going to be as the main plant during the concatenate operation.
                leaf_base_concate = batch_leaf_base.iloc[rank_id * 6 : rank_id * 6 + 3, idx[selected_pc_id_of_leaf[0]]].values  # Leaf base of the leaf which is going to be concatenated to the plant.
                orientation_vec_this = batch_leaf_base.iloc[rank_id * 6 + 3 : rank_id * 6 + 6, idx[pc_id]].values
                orientation_vec_concate = batch_leaf_base.iloc[rank_id * 6 + 3 : rank_id * 6 + 6, idx[selected_pc_id_of_leaf[0]]].values
                # Rotate the leaf being concatenated to make its main axis 
                # overlap with the original leaf.
                ori_vec_01 = np.zeros((1, 3), dtype="float32")
                ori_vec_02 = np.zeros((1, 3), dtype="float32")
                ori_vec_01[0, 0:2] = orientation_vec_this[0:2].reshape([1, -1])
                ori_vec_02[0, 0:2] = orientation_vec_concate[0:2].reshape([1, -1])
                cosval = np.dot(ori_vec_01[0, 0:2], ori_vec_02[0, 0:2])  # Both orientation vectors have a normal of 1.
                cross_product = np.cross(ori_vec_02, ori_vec_01)
                sinval = np.sqrt(1 - np.power(cosval, 2)) * (cross_product[0, -1]/abs(cross_product[0, -1]))  # The last term in the bracket is used to determine whether the value of sin() is positive (or negative).
                rotation_matrix = np.array([[cosval, sinval, 0],
                                            [-sinval, cosval, 0],
                                            [0, 0, 1]])
                leaf_feature_concate = copy.deepcopy(leaf_feature_pool[selected_pc_id_of_leaf[0]])
                concate_leaf_base_xy0 = np.tile(np.array([[leaf_base_concate[0], leaf_base_concate[1], 0]]), \
                                               (leaf_feature_concate.shape[0], 1))
                pc_z_centered = leaf_feature_concate[:, 0:3] - concate_leaf_base_xy0  # Move the leaf being concatenated to the main plant to the origin point of the axis, in order to conduct further rotation.
                shift_info = np.tile(np.array([[leaf_base_this[0], leaf_base_this[1], leaf_base_this[2] - leaf_base_concate[2]]]), \
                                    (leaf_feature_concate.shape[0], 1))
                leaf_feature_concate[:, 0:3] = np.dot(pc_z_centered, rotation_matrix) + shift_info
                
                pc_leaf_crossovered = np.concatenate((leaf_feature_concate, \
                                                      leaf_instance_label_pool[selected_pc_id_of_leaf[0]], \
                                                      leaf_semantic_label_pool[selected_pc_id_of_leaf[0]]), axis=1)
                
                all_raw_batch_pcs[idx[pc_id]] = np.concatenate((pc_this, pc_leaf_crossovered), axis=0)
    except:
        print("ERROR!!!")
        print(batch_pc_file_name)
        print("Cultivar: ", cultivar)
        print("Range of leaf ranks for crossover: ", [crossover_start_rank, crossover_end_rank])
        print("Selected crossover ranks: ", rank_crossover)
        os._exit()
    
    return all_raw_batch_pcs
