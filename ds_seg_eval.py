import datetime
import argparse
import importlib
import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import show3d_balls
import h5py
from pathlib import Path
import open3d as o3d
from scipy.spatial import KDTree
import copy


parser = argparse.ArgumentParser()

# GPU & model selection
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: model]')
parser.add_argument('--check_point_path', default='./log', help='Check point path')
parser.add_argument('--repeat_id_best', type=int, default=4, help='Model ID with the best performance [default: 0]')
parser.add_argument('--nsampling', type=int, default=1, help='Number of down sampling times in order to get a integrated stemwork extraction result [default: 5]')
parser.add_argument('--seg_thresh', type=float, default=0.90, help='Probability threshold for a point is considered as a stemwork point [default: 0.9]')
parser.add_argument('--pn_origin', type=bool, default=False, help='Whether to use the original number of S&G (feature propagation) layers in training [default: True]')

# Test set parameters
parser.add_argument('--deploy_file_path', default='./seg_results_for_deployment/source_plant_pcs/surface_pcs_with_semantic_instance_labels (refined) (45 pcs).h5', help='Path of the point cloud to be deployed')
parser.add_argument('--inner_removal', type=bool, default=False, help='Whether to remove the inner points with in the object point cloud [default: False]')
parser.add_argument('--normalisation_enable', type=bool, default=True, help='Whether to conduct the normalisation procedure to the target point cloud [default: False]')
parser.add_argument('--down_sampling_strategy', default='global_random', help='The method to down sample the original point cloud into a uniform-size point cloud')
parser.add_argument('--num_classes', type=int, default=4, help='Number of semantic classes [default: 4]')
parser.add_argument('--stemwork_class_id_rearranged', type=int, default=2, help='The index for the stemwork class (after using get_dataset Fcn) [default: 2]')
parser.add_argument('--stemwork_class_id_origin', type=int, default=3, help='The index for the stemwork class (originally being labeled) [default: 3]')
parser.add_argument('--num_point', type=int, default=50000, help='Point Number [default: 2048]')
parser.add_argument('--num_features', type=int, default=51, help='Demension of point features including positional info [default: 51]')
parser.add_argument('--test_file_list_path', default='./dataset/test_file_list.txt', help='Path of the point cloud to be deployed')

# Output directory
parser.add_argument('--output_dir', default='./seg_results_for_deployment', help='Output dir [default: ./test_results]')

FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
CHECK_POINT_PATH = FLAGS.check_point_path
selected_model_id = FLAGS.repeat_id_best
NSAMPLING = FLAGS.nsampling
SEG_THRESH = FLAGS.seg_thresh
PN_ORIGIN = FLAGS.pn_origin
DEPLOY_FILE_PATH = FLAGS.deploy_file_path
INNER_REMOVAL = FLAGS.inner_removal
NORMALISATION_ENABLE = FLAGS.normalisation_enable
DOWM_SAMPLING_STRATEGY = FLAGS.down_sampling_strategy
NUM_CLASSES=FLAGS.num_classes
STEMWORK_CLASS_ID_REARRANGED = FLAGS.stemwork_class_id_rearranged
STEMWORK_CLASS_ID_ORIGIN = FLAGS.stemwork_class_id_origin
NUM_POINT = FLAGS.num_point
NUM_FEATURES = FLAGS.num_features
TEST_FILE_LIST_PATH = FLAGS.test_file_list_path
OUTPUT_DIR = FLAGS.output_dir

MODEL = importlib.import_module(FLAGS.model) # import network module


def print_log(msg, stream=None):
    formatted_msg = "[%s] %s" % (str(datetime.datetime.now()), msg)
    print(formatted_msg)
    if stream is not None:
        stream.write(formatted_msg)


def get_model(model_path, batch_size, num_point):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(batch_size, NUM_POINT, NUM_FEATURES)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            if PN_ORIGIN:
                pred, end_points = MODEL.get_model_origin(pointclouds_pl, \
                                       is_training_pl, NUM_FEATURES, NUM_CLASSES)
            else:
                pred, end_points = MODEL.get_model_simplify(pointclouds_pl, \
                                       is_training_pl, NUM_FEATURES, NUM_CLASSES)
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, model_path)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred}
        return sess, ops
    

def inference(sess, ops, pc, batch_size):
    ''' pc: BxNx3 array, return BxN pred '''
    assert pc.shape[0]%batch_size == 0
    num_batches = pc.shape[0]//batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    for i in range(num_batches):
        feed_dict = {ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
                     ops['is_training_pl']: False}
        batch_logits = sess.run(ops['pred'], feed_dict=feed_dict)
        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
    return logits


def get_pc_features(pc):
    """
    If the original point cloud being deployed do not contain normal features
    or has not been normalised, conduct corresponding procedure using this
    function.
    """
    # Zero centering & normalisation
    center = np.mean(pc[:, 0:3], axis=0)
    pc_centered = pc[:, 0:3] - center
    bound_max = np.max(pc_centered, axis=0)
    bound_min = np.min(pc_centered, axis=0)
    # span = bound_max - bound_min
    # id_axis = np.where(span == np.max(span))[0].tolist()
    selected_span = np.max([abs(bound_max[2]), abs(bound_min[2])])
    scale = 1/selected_span
    pc_pos_normalised = pc_centered * scale
    
    normalisation_info = np.concatenate((center.reshape([1, -1]), \
                                         scale.reshape([1, -1])), axis=1)
    
    # Normals of points
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc_pos_normalised[:, 0:3])
    pc_o3d.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))  # Tune these two parameters carefully, or it may lead to a all-upward normal estimation
    normal_array = np.asarray(pc_o3d.normals)
    
    # Normalisation of colours
    pc_color_normalised = pc[:, 3:48]/255
    
    pc = np.c_[pc_pos_normalised[:, 0:3], normal_array, pc_color_normalised]  # [x, y, z, norm_x, norm_y, norm_z, colour (3*15)]
    
    return pc, normalisation_info.squeeze()


def softmax(x, axis=1):
    row_max = x.max(axis=axis)
    row_max=row_max.reshape(-1, 1)
    x = x - row_max
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def get_surface_points(original_labeled_pc, res=1):
    points = original_labeled_pc[:, 0:3]
    # colors = original_labeled_pc[:, 3:]
    tree = KDTree(points)
    
    nn_radius = res + 0.1

    surface_point_id = []    
    for index, point in enumerate(points):
        neighbors = tree.query_ball_point(point, nn_radius)
        
        if len(neighbors) <= 6:
            surface_point_id.append(index)
    
    pc_surface = original_labeled_pc[surface_point_id, :]

    return pc_surface


def get_block_wise_sampling_idx(original_num_point):
    block_size  = int(original_num_point / NUM_POINT)
    space = np.tile(np.array(list(range(0, NUM_POINT*block_size, block_size))).reshape([1, -1]), (block_size, 1))
    for block_id in range(NUM_POINT):
        idx = np.random.choice(range(block_size), size=block_size, replace=False).reshape([-1, 1])
        if block_id == 0:
            pt_selection_sequence = idx
        else:
            pt_selection_sequence = np.concatenate((pt_selection_sequence, idx), axis=1)
    pt_selection_sequence += space
    return pt_selection_sequence, block_size


def get_global_random_sampling_idx(original_num_point):
    for sampling_idx in range(NSAMPLING):
        idx = np.random.choice(range(original_num_point), size=NUM_POINT, replace=False).reshape([1, -1])
        if sampling_idx == 0:
            pt_selection_sequence = idx
        else:
            pt_selection_sequence = np.concatenate((pt_selection_sequence, idx), axis=0)
    return pt_selection_sequence, NSAMPLING


if __name__ == '__main__':
    
    ## - Data Loading
    f = h5py.File(DEPLOY_FILE_PATH, "r")
    
    test_file_list = pd.read_csv(TEST_FILE_LIST_PATH, header=None)
    
    f1_rec = list()
    iou_rec = list()
    
    n_finished = 0
    for pc_csv_name in test_file_list.values:
        
        pc_real_name = pc_csv_name[0][0 : pc_csv_name[0].find(".")]
        
        n_finished += 1
        print("Currently dealing with: ", n_finished, ", ", pc_real_name)
    
        pc_deploy = f.get(pc_real_name)[()]
    
        if INNER_REMOVAL:
            pc_deploy = get_surface_points(pc_deploy)
        
        if NORMALISATION_ENABLE:
            # If the target point cloud has not been normalised, set this variable
            # as True. The normalised point info is only used for the network
            # prediction. Final output point clouds with predicted labels will be
            # presented by using the non-normalised point info.
            pc_normalised, _ = get_pc_features(pc_deploy)  # Only used for network prediction
        else:
            # If the target point cloud has been normalised, set this variable as
            # False. The normalised point info will be restored to normal coordinate
            # system and normal colour space (0-255) before the final output with 
            # predicted labels is presented.
            pc_normalised = copy.deepcopy(pc_deploy[:, 0:NUM_FEATURES])  # Only used for network prediction
            normalisation_info = pd.read_csv(os.path.join(os.path.dirname(DEPLOY_FILE_PATH), "test_data_normalisation_info.csv"), \
                                             index_col=0)
            center = normalisation_info.loc[PC_NAME_IN_H5, "center_x":"center_z"].values
            scale = normalisation_info.loc[PC_NAME_IN_H5, "scaleNSAMPLING"]
            pc_deploy[:, 0:3] = pc_deploy[:, 0:3] / scale + np.tile(center.reshape([1, -1]), (NUM_POINT, 1))
        
        ckptstate = tf.train.get_checkpoint_state(os.path.join(CHECK_POINT_PATH, "repeat_" + str(selected_model_id)))
        model_path = ckptstate.model_checkpoint_path
        
        ## - Point Cloud Down Sampling
        # Random Shuffle
        rand_shuffle_idx = np.random.choice(range(pc_deploy.shape[0]), size=pc_deploy.shape[0], replace=False)
        pc_deploy = pc_deploy[rand_shuffle_idx, :]
        pc_normalised = pc_normalised[rand_shuffle_idx, :]
        
        # Select a down sampling strategy and get a down sampled point cloud
        if DOWM_SAMPLING_STRATEGY == "block_wise":
            pt_selection_sequence, num_resampling = get_block_wise_sampling_idx(pc_deploy.shape[0])
        else:
            pt_selection_sequence, num_resampling = get_global_random_sampling_idx(pc_deploy.shape[0])
        
        ## - Annotation Prediction
        stemwork_pts_confirmed = []
        for sampling_id in range(num_resampling):
            
            selected_idx = pt_selection_sequence[sampling_id, :]
            
            ps = pc_normalised[selected_idx, :]  # Used for network prediction only.
            
            sess, ops = get_model(model_path, batch_size=1, num_point=ps.shape[0])
            segp_logits = inference(sess, ops, np.expand_dims(ps, 0), batch_size=1)
            segp_logits = segp_logits.squeeze()
            
            segp_prob = softmax(segp_logits, axis=1)
            
            idx = np.where(segp_prob[:, STEMWORK_CLASS_ID_REARRANGED] >= SEG_THRESH)[0].tolist()
            # idx = np.where(np.argmax(segp_prob, axis=1) == STEMWORK_CLASS_ID_REARRANGED)[0].tolist()
            stemwork_pt_candidates_in_selected_idx = selected_idx[idx]
            stemwork_pt_candidates = pc_deploy[stemwork_pt_candidates_in_selected_idx, :]
            
            pt_idx_confirmed_this_sampling_attempt = []
            for k in range(stemwork_pt_candidates.shape[0]):
                if stemwork_pt_candidates_in_selected_idx[k] not in stemwork_pts_confirmed:
                    pt_idx_confirmed_this_sampling_attempt.append(k)
                    stemwork_pts_confirmed.append(stemwork_pt_candidates_in_selected_idx[k])
            
        print("Number of decided stemwork points: ", len(stemwork_pts_confirmed))
        
        ## Evaluate the down sampling + the stemwork segmentation performance
        pred_mask = np.zeros((pc_deploy.shape[0],), dtype="int8")
        pred_mask[stemwork_pts_confirmed] = 1
        real_mask = np.array(pc_deploy[:, -1] == STEMWORK_CLASS_ID_ORIGIN, dtype="int8")
        TP = np.sum((pred_mask == 1) & (real_mask == 1))
        FP = np.sum((pred_mask == 1) & (real_mask == 0))
        FN = np.sum((pred_mask == 0) & (real_mask == 1))
        if TP + FP != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        if TP + FN != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            
        iou = TP / (TP + FP + FN)
        
        iou_rec.append(iou)
        f1_rec.append(f1)
    
    print(f1_rec)
    print("The average down sampling + segmentation F1 score: ", np.mean(np.array(f1_rec)))
    print("The average down sampling + segmentation IoU score: ", np.mean(np.array(iou_rec)))
        
    f.close()
    