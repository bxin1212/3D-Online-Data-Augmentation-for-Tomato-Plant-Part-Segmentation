#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:42:52 2021

@author: bxin
"""

import numpy as np
import open3d as o3d
import os


def load_obj_file(objFilePath):
    with open(objFilePath) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3]), \
                               float(strs[4]), float(strs[5]), float(strs[6])))
            if strs[0] == "vt":
                break
    return np.array(points)


def visualisation(pc_array):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_array[:, 0:3])
    if pc_array.shape[1] == 6:
        pc.colors =  o3d.utility.Vector3dVector(pc_array[:, 3:6])
    o3d.visualization.draw_geometries([pc])


if __name__ == "__main__":
    
    pc_file_path = "./test_results/repeat_0/pred_0.obj"
    pc_array = load_obj_file(pc_file_path)
    
    visualisation(pc_array)