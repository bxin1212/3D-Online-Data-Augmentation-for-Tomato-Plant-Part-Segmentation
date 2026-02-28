#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:39:31 2022

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
                isTestset=True, \
                enhance_rate=1.0):
    
    # Load all point clouds
    f = h5py.File(file_path, "r")
    
    i = 0
    
    for filename in f.keys():
        
        pc_raw = f.get(filename)[()]  # [x, y, z, norm_x, norm_y, norm_z, colour (3*15), ins_labels, sem_labels]
    
        # Get colour channels
        pc_r_median = np.median(pc_raw[:, np.arange(3, pc_raw.shape[1], 3)], axis=1) * 3
        pc_r_median[pc_r_median > 1] = 1
        pc_g_median = np.median(pc_raw[:, np.arange(4, pc_raw.shape[1], 3)], axis=1) * 3
        pc_g_median[pc_g_median > 1] = 1
        pc_b_median = np.median(pc_raw[:, np.arange(5, pc_raw.shape[1], 3)], axis=1) * 3
        pc_b_median[pc_b_median > 1] = 1
        pc_color_normalised = np.concatenate((pc_r_median.reshape([-1, 1]), \
                                              pc_g_median.reshape([-1, 1]), \
                                              pc_b_median.reshape([-1, 1])), axis=1)
        
        pc_normal_color_label = np.c_[pc_raw[:, 0:3], \
                                      pc_raw[:, 3:6], \
                                      pc_color_normalised, \
                                      pc_raw[:, -2], \
                                      pc_raw[:, -1]]  # [x, y, z, norm_x, norm_y, norm_z, r_med, g_med, b_med, ins_labels, sem_labels]
        
        if i == 0:
            mode = "w"
        else:
            mode = "a"
        
        save_h5_file_full_path = os.path.join(save_path, save_h5_file_name)
        
        save_dataset(save_h5_file_full_path, mode, filename, pc_normal_color_label.astype("float32"))
        
        i += 1
        
        print(i, filename)
        
    f.close()


def save_dataset(save_path, mode, save_key, data):
    f_save = h5py.File(save_path, mode)
    f_save.create_dataset(save_key, data=data)
    f_save.close()


if __name__ == '__main__':
    
    isTestset = False
    
    file_path = "./dataset/inner_point_removal/Pos_Norm_Color_Sem Ins_Test DS (45 pcs)/training_set.h5"
    save_h5_file_name = "test_set.h5" if isTestset else "training_set.h5"
    save_path = "./dataset"
    
    enhance_rate = 3.0  # To enhance the colour contrast
    
    get_dataset(file_path=file_path, \
                save_path=save_path, \
                save_h5_file_name=save_h5_file_name, \
                isTestset=isTestset, \
                enhance_rate=enhance_rate)