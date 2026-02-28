#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 18:10:47 2022

@author: bxin
"""

import numpy as np
import open3d as o3d
import os
from matplotlib import pyplot as plt
import copy


def get_pt_density_mat(pc_raw, radius_thresh):
    density_mat = list()
    for i in range(pc_raw.shape[0]):
        target_pt_coordinate = pc_raw[i, 0:3]
        dist = np.sqrt(np.sum(np.power(pc_raw[:, 0:3] - np.tile(target_pt_coordinate, (pc_raw.shape[0], 1)), 2), axis=1))
        density_mat.append(np.sum(dist <= radius_thresh))
    return np.array(density_mat)


def pc_density_filter(pc_raw, density_mat, thresh_method="manual_specify", mode="more", density_thresh=70):
    """
    Parameters
    ----------
    pc_raw: Numpy arrays with dimension NUM_POINTS-by-NUM_FEATURES
        DESCRIPTION:
    density_mat: TYPE
        DESCRIPTION:
    thresh_method: TYPE, options: "manual_specify" & "otsu"
        DESCRIPTION: The default is "manual_specify".
    mode: TYPE, options: "more" & "less"
        DESCRIPTION:
    density_thresh: TYPE, optional
        DESCRIPTION: The default is 70.

    Returns
    -------
    density_thresh: TYPE
        DESCRIPTION:
    selected_idx: TYPE
        DESCRIPTION:
    """
    # Otsu's thresholding
    if thresh_method == "otsu":
        current_best = 0
        density_thresh = np.min(density_mat)
        for current_thresh in range(np.min(density_mat), np.max(density_mat) + 1):
            foreground = density_mat[density_mat >= current_thresh]
            background = density_mat[density_mat < current_thresh]
            w0 = foreground.shape[0] / density_mat.shape[0]
            if foreground.shape[0] != 0:
                foreground_expectation = np.mean(foreground)
            else:
                foreground_expectation = 0
            w1 = background.shape[0] / density_mat.shape[0]
            if background.shape[0] != 0:
                background_expectation = np.mean(background)
            else:
                background_expectation = 0
            between_class_var = w0 * w1 * np.power(foreground_expectation - background_expectation, 2)
            if between_class_var > current_best:
                density_thresh = current_thresh
                current_best = between_class_var
    selected_idx = density_mat >= density_thresh if mode=="more" else density_mat <= density_thresh
    return pc_raw[selected_idx, :], density_thresh, selected_idx


def pt_density_visualisation(pc_raw, density_mat, colour_map):
    colour_mat = colour_map(list(density_mat))[:, 0:3]
    pc_visual = o3d.geometry.PointCloud()
    pc_visual.points = o3d.utility.Vector3dVector(pc_raw[:, 0:3])
    pc_visual.colors = o3d.utility.Vector3dVector(colour_mat)
    o3d.visualization.draw_geometries([pc_visual])


if __name__ == "__main__":
    
    plant_idx = "Harvest_01_PotNr_145"
    
    noise_removal = True
    
    # File Path
    work_path = "/home/bxin/Professional/Plant_Organ_Seg/PointNet++ Based Plant Organ Semantic Segmentation/seg_results_for_deployment/results/block_wise_random_down_sampling/with_argmax"
    file_path = os.path.join(work_path, plant_idx, "combined_stemwork.csv")
    if noise_removal:
        save_path = os.path.join(work_path, plant_idx, "cleaned_stemwork_with_noise_removal.csv")
    else:
        save_path = os.path.join(work_path, plant_idx, "cleaned_stemwork_without_noise_removal.csv")
    
    ## - Parameter Settings
    density_radius = 2
    
    density_thresh_noise_removal = 3
    density_thresh_first_branch_removal = 50
    
    dbscan_eps = 20
    dbscan_npts = 90
    
    branch_removal_thresh = 0.07
    
    pc_raw = np.loadtxt(file_path, delimiter=",")
    
    ## - Remove the first leaf branch at the bottom, which is usually 
    ## disconnected with the main plant. This is based on the horizontal
    ## density map of stemwork points.
    pc_bottom_coordinate = np.min(pc_raw, axis=0)
    pc_top_coordinate = np.max(pc_raw, axis=0)
    
    # - Step 1: Calculate point density over the projections on XOY plane.
    projection_xoy = np.concatenate((pc_raw[:, 0:2], \
                                     np.zeros((pc_raw.shape[0], 1), dtype="float16")), axis=1)
    horizontal_density_mat = get_pt_density_mat(projection_xoy, density_radius)
    colour_map = plt.cm.get_cmap('hot', np.max(horizontal_density_mat))
    pt_density_visualisation(pc_raw, horizontal_density_mat, colour_map)
    
    # - Step 2: Select the points whose density is larger than the threshold
    pc_stem_removal, density_thresh, selected_idx = pc_density_filter(pc_raw, \
                                                                        horizontal_density_mat, \
                                                                        thresh_method="manual_specify", \
                                                                        mode="less", \
                                                                        density_thresh=density_thresh_first_branch_removal)
    pc_visual = o3d.geometry.PointCloud()
    pc_visual.points = o3d.utility.Vector3dVector(pc_stem_removal[:, 0:3])
    o3d.visualization.draw_geometries([pc_visual])
    
    # - Step 3: Instance segmentation with DBSCAN
    pc_dbscan = o3d.geometry.PointCloud()
    pc_dbscan.points = o3d.utility.Vector3dVector(pc_stem_removal[:, 0:3])
    labels = np.array(pc_dbscan.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_npts))
    max_label = labels.max()
    nr_clusters = max_label + 1
    print('Number of clusters', nr_clusters)
    
    # Show the clustered point cloud
    colors = plt.get_cmap("tab20")(labels/(max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    objects_pcd_show = copy.deepcopy(pc_dbscan)
    objects_pcd_show.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([objects_pcd_show])
    
    # Get the objects as individual point clouds
    n_existing_instance = 0
    n_removed_instance = 0
    for i in range(-1, nr_clusters):
        obj_point_ids = np.where(labels == i)[0]
        instance_pts = pc_dbscan.select_by_index(obj_point_ids)
        # Get the first 10% points with lowest Z coordinates, and calculate the
        # centroid of these selected points. If the distance between this centroid
        # and the bottom of the point cloud is smaller than a given threshold,
        # remove this leaf candidate.
        instance_centroid = instance_pts.get_center()
        z_dist = np.abs(instance_centroid[2] - pc_bottom_coordinate[2])
        print(z_dist)
        if z_dist >= branch_removal_thresh * np.abs(pc_top_coordinate[2] - pc_bottom_coordinate[2]):
            if n_existing_instance == 0:
                pc_branch_removal = np.asarray(instance_pts.points)
            else:
                pc_branch_removal = np.concatenate((pc_branch_removal, np.asarray(instance_pts.points)), axis=0)
            n_existing_instance += 1
        else:
            if  n_removed_instance == 0:
                pc_removed = np.asarray(instance_pts.points)
            else:
                pc_removed = np.concatenate((pc_removed, np.asarray(instance_pts.points)), axis=0)
            n_removed_instance += 1
    not_selected_idx = [False if idx == True else True for idx in selected_idx]
    pc_branch_removal = np.concatenate((pc_branch_removal, pc_raw[not_selected_idx, 0:3]), axis=0)  # Include the stem points
    
    # Visualisation
    if n_removed_instance > 0:
        pc_visual_numpy = np.concatenate((pc_branch_removal, pc_removed), axis=0)
        pc_colours_removal = np.zeros((pc_branch_removal.shape[0], 3), dtype="float16")
        pc_colours_removal[:, 0] = 0
        pc_colours_removal[:, 1] = 0
        pc_colours_removal[:, 2] = 1
        pc_colours_removed = np.zeros((pc_removed.shape[0], 3), dtype="float16")
        pc_colours_removed[:, 0] = 1
        pc_colours_removed[:, 1] = 0
        pc_colours_removed[:, 2] = 0
        pc_colours = np.concatenate((pc_colours_removal, pc_colours_removed), axis=0)
    else:
        pc_visual_numpy = pc_branch_removal
        pc_colours = np.zeros((pc_branch_removal.shape[0], 3), dtype="float16")
        pc_colours[:, 0] = 0
        pc_colours[:, 1] = 0
        pc_colours[:, 2] = 1
    pc_visual = o3d.geometry.PointCloud()
    pc_visual.points = o3d.utility.Vector3dVector(pc_visual_numpy[:, 0:3])
    pc_visual.colors = o3d.utility.Vector3dVector(pc_colours[:, 0:3])
    o3d.visualization.draw_geometries([pc_visual])
    
    
    ## - Remove the noise points with local point density map
    if noise_removal:
        local_density_mat = get_pt_density_mat(pc_branch_removal, density_radius)
        
        colour_map = plt.cm.get_cmap('hot', np.max(local_density_mat))
        pt_density_visualisation(pc_branch_removal, local_density_mat, colour_map)
        
        pc_filtered, density_thresh, _ = pc_density_filter(pc_branch_removal, \
                                                        local_density_mat, \
                                                        thresh_method="manual_specify", \
                                                        mode="more", \
                                                        density_thresh=density_thresh_noise_removal)
        
        pc_visual = o3d.geometry.PointCloud()
        pc_visual.points = o3d.utility.Vector3dVector(pc_filtered[:, 0:3])
        o3d.visualization.draw_geometries([pc_visual])
    else:
        pc_filtered = pc_branch_removal
    
    
    np.savetxt(save_path, pc_filtered, delimiter=",")
    
    
    # Show the objects one by one
    # for object_pcd in object_pcds:
    #     o3d.visualization.draw_geometries([object_pcd])
    