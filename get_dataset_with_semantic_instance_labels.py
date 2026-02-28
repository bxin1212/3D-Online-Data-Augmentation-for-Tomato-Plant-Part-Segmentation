#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:17:19 2021

@author: bxin
"""

import h5py
import numpy as np
import open3d as o3d
import os
import pandas as pd


def get_dataset(file_path, \
                save_path, \
                save_h5_file_name, \
                isDown_sampling, \
                test_file_list, \
                isTestset=True, \
                num_points=50000):
    
    # Load all point clouds
    f = h5py.File(file_path, "r")
    
    i = 0
    
    if isTestset:
        test_dataset_info = np.zeros((test_file_list.shape[0], 4), dtype="float32")
        test_file_name = [] 
    
    for filename in f.keys():
        
        if filename + ".csv" in test_file_list.values:
            if not isTestset:
                continue
            
        if filename + ".csv" not in test_file_list.values:
            if isTestset:
                continue
        
        pc_raw = f.get(filename)[()]  # [x, y, z, colors (3*15), labels]
        
        # Check whether there is any NaN values
        id_nan = np.where(np.isnan(pc_raw))[0]
        
        if len(id_nan) != 0:
            print("----- WARNING! NaN coordinates found in the following point cloud file! -----")
            print(filename)
        
        """
        Original point label description:
        For tomato plant semantic segmentation dataset, points were labeled as:
            0 - Soil basis
            1 - Stick
            2 - Unclassified (has been removed by preceding code)
            3 - Stemwork
            4 - Other organs
            5 - Trusses (not all point clouds have this class, the first batch of labeling do not have this class)
            6 - Side shoots (not all point clouds have this class)
        Since points belonging to label 2 (unclassified) have been removed from
        the dataset, points labeled with 3 and 4 should be re-arranged to 2 and 
        3 respectively.
        """
        
        # Remove points belonging to "Unclassified" & "trusses"
        exclude_id = np.where((pc_raw[:, -1] == 2) | (pc_raw[:, -1] == 5) | (pc_raw[:, -1] == 6))
        pc_excluded = np.delete(pc_raw, exclude_id, axis=0)
        
        # Re-arrange labels with respect to individual points
        ids = np.where(pc_excluded[:, -1] == 3)
        pc_excluded[ids, -1] = 2
        ids = np.where(pc_excluded[:, -1] == 4)
        pc_excluded[ids, -1] = 3
        
        if isTestset | isDown_sampling:
            # Random down sampling
            selected_id = np.random.choice(range(pc_excluded.shape[0]), \
                                           size=num_points, \
                                           replace=None, \
                                           p=None)
            pc_excluded = pc_excluded[selected_id,]
        
        # Zero centering & normalisation
        center = np.mean(pc_excluded[:, 0:3], axis = 0)
        pc_centered = pc_excluded[:, 0:3] - center
        bound_max = np.max(pc_centered, axis = 0)
        bound_min = np.min(pc_centered, axis = 0)
        span = bound_max - bound_min
        id_axis = np.where(span == np.max(span))[0].tolist()
        selected_span = np.max([abs(bound_max[id_axis]), abs(bound_min[id_axis])])
        scale = 1/selected_span
        pc_pos_normalised = pc_centered * scale
        
        if isTestset:
            test_dataset_info[i, 0:3] = center
            test_dataset_info[i, 3] = scale
            test_file_name.append(filename)
        
        # Distance to the centroid axis
        centroid = np.mean(pc_pos_normalised, axis=0)
        dist = np.sqrt(np.sum(np.square(pc_pos_normalised[:, 0:2] - centroid[0:2]), axis=1))
        
        # Get colour channels
        # pc_r_median = np.median(pc_excluded[:, np.arange(3, pc_excluded.shape[1], 3)], axis=1) / 255
        # pc_g_median = np.median(pc_excluded[:, np.arange(4, pc_excluded.shape[1], 3)], axis=1) / 255
        # pc_b_median = np.median(pc_excluded[:, np.arange(5, pc_excluded.shape[1], 3)], axis=1) / 255
        # pc_color_normalised = np.concatenate((pc_r_median.reshape([-1, 1]), \
        #                                       pc_g_median.reshape([-1, 1]), \
        #                                       pc_b_median.reshape([-1, 1])), axis=1) * 2
        
        pc_color_normalised = pc_excluded[:, 3:48]/255
        
        pc_normalised = np.concatenate((pc_pos_normalised, pc_color_normalised, dist.reshape(-1, 1)), axis=1)
        
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_normalised[:, 0:3])
        pc.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, \
                                                                                max_nn=10))  # Tune these two parameters carefully, or it may lead to a all-upward normal estimation
        normal_array = np.asarray(pc.normals)
        
        pc_normal_color_label = np.c_[pc_normalised[:, 0:3], \
                                      normal_array, \
                                      #pc_normalised[:, 48], \
                                      pc_normalised[:, 3:48], \
                                      pc_excluded[:, -2], \
                                      pc_excluded[:, -1]]  # [x, y, z, norm_x, norm_y, norm_z, colors (3*15), labels]
        # pc_normal_color_label = np.c_[pc_normalised[:, 0:3], \
        #                               normal_array, \
        #                               #pc_normalised[:, 48], \
        #                               pc_normalised[:, 3:6], \
        #                               pc_excluded[:, -1]]  # [x, y, z, norm_x, norm_y, norm_z, colors (3*15), labels]
        
        if i == 0:
            mode = "w"
        else:
            mode = "a"
        
        save_h5_file_full_path = os.path.join(save_path, save_h5_file_name)
        
        save_dataset(save_h5_file_full_path, mode, filename, pc_normal_color_label.astype("float32"))
        
        i += 1
        
        print(i, filename)
    
    # Save centering & normalisation information for the test dataset
    if isTestset:
        df = pd.DataFrame(test_dataset_info, \
                          columns=["center_x", "center_y", "center_z", "scale"], \
                          index=test_file_name)
        df.to_csv(os.path.join(save_path, "test_data_normalisation_info.csv"))
        
    f.close()


def save_dataset(save_path, mode, save_key, data):
    f_save = h5py.File(save_path, mode)
    f_save.create_dataset(save_key, data=data)
    f_save.close()


if __name__ == '__main__':
    
    isTestset = False
    isDown_sampling = True
    NUM_POINTS = 50000
    
    file_path = "../Original Labeled Point Cloud Dataset/without inner points/surface_pcs_with_semantic_instance_labels (refined) (45 pcs).h5"
    save_h5_file_name = "test_set.h5" if isTestset else "training_set.h5"
    save_path = "./dataset"
    test_file_list_path = "./dataset/test_file_list.txt"
    
    test_file_list = pd.read_csv(test_file_list_path, header=None)
    
    get_dataset(file_path=file_path, \
                save_path=save_path, \
                save_h5_file_name=save_h5_file_name, \
                test_file_list=test_file_list, \
                isDown_sampling=isDown_sampling, \
                isTestset=isTestset, \
                num_points=NUM_POINTS)