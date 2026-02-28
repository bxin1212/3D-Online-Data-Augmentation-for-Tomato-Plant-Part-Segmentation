import datetime
import argparse
import importlib
import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import cm
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import show3d_balls
import h5py
from pathlib import Path


parser = argparse.ArgumentParser()

# GPU & model selection
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: model]')
parser.add_argument('--check_point_path', default='./log', help='Check point path')
parser.add_argument('--num_repeats', type=int, default=10, help='Number of training times for a single network configuration and dataset [default: 5]')
parser.add_argument('--pn_origin', type=bool, default=False, help='Whether to use the original number of S&G (feature propagation) layers in training [default: True]')

# Test set parameters
parser.add_argument('--dataset_path', default='./dataset/inner_point_removal/Pos_Norm_Color_Sem Ins_Test DS (45 pcs)/test_set.h5', help='Dataset path')
parser.add_argument('--num_classes', type=int, default=4, help='Number of semantic classes [default: 4]')
parser.add_argument('--num_point', type=int, default=50000, help='Point Number [default: 2048]')
parser.add_argument('--num_features', type=int, default=51, help='Demension of point features including positional info [default: 51]')

# Output directory
parser.add_argument('--output_dir', default='./test_results', help='Output dir [default: ./test_results]')

FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
CHECK_POINT_PATH = FLAGS.check_point_path
NUM_REPEATS = FLAGS.num_repeats
PN_ORIGIN = FLAGS.pn_origin
DATASET_PATH = FLAGS.dataset_path
NUM_CLASSES=FLAGS.num_classes
NUM_POINT = FLAGS.num_point
NUM_FEATURES = FLAGS.num_features
OUTPUT_DIR = FLAGS.output_dir

MODEL = importlib.import_module(FLAGS.model) # import network module

TEST_DATASET = h5py.File(DATASET_PATH, "r")
normalisation_info = pd.read_csv(os.path.join(os.path.dirname(DATASET_PATH), "test_data_normalisation_info.csv"))
normalisation_info = normalisation_info.set_index("Unnamed: 0")


def print_log(msg, stream=None):
    formatted_msg = "[%s] %s" % (str(datetime.datetime.now()), msg)
    print(formatted_msg)
    if stream is not None:
        stream.write(formatted_msg)


def output_color_point_cloud(data, seg, out_file, color_map):
    with open(out_file, 'w') as f:
        for i in range(len(seg)):
            color = color_map(int(seg[i]))
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))


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
    return np.argmax(logits, 2)


if __name__ == '__main__':
    
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    
    color_map = cm.get_cmap('viridis', NUM_CLASSES)
    SIZE = len(TEST_DATASET.keys())
        
    f1_rec = np.zeros((NUM_REPEATS, 3, NUM_CLASSES), dtype="float32")
    f1_stemwork_on_stem_rec = np.zeros((NUM_REPEATS, 1), dtype="float32")
    f1_stemwork_on_levels_rec = np.zeros((NUM_REPEATS, 10), dtype="float32")
    f1_stemwork_cultivar_rec = np.zeros((NUM_REPEATS, 3), dtype="float32")
    iou_rec = np.zeros((NUM_REPEATS, NUM_CLASSES), dtype="float32")
    for repeat_id in range(NUM_REPEATS):
        
        ckptstate = tf.train.get_checkpoint_state(os.path.join(CHECK_POINT_PATH, "repeat_" + str(repeat_id)))
        model_path = ckptstate.model_checkpoint_path
        
        save_folder = os.path.join(OUTPUT_DIR, "repeat_" + str(repeat_id))
        my_file = Path(save_folder)
        if not my_file.is_dir(): os.mkdir(save_folder)
    
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_pred_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        
        total_seen_stemwork_on_stem = 0
        total_pred_stemwork_on_stem = 0
        total_correct_stemwork_on_stem = 0
        
        total_seen_stemwork_on_levels = [0 for _ in range(10)]
        total_pred_stemwork_on_levels = [0 for _ in range(10)]
        total_correct_stemwork_on_levels = [0 for _ in range(10)]
        
        total_seen_stemwork_cultivar = [0 for _ in range(3)]
        total_pred_stemwork_cultivar = [0 for _ in range(3)]
        total_correct_stemwork_cultivar = [0 for _ in range(3)]
    
        for i in range(SIZE):
            print_log(">>>> repeat_id " + str(repeat_id+1) + "/" + str(NUM_REPEATS) + \
                      " ------ running sample " + str(i+1) + "/" + str(SIZE))
            
            pc_name = normalisation_info.index.values[i]
            
            pc = TEST_DATASET.get(pc_name)[()]
            
            ps = pc[:, 0:NUM_FEATURES]
            seg = pc[:, -1]
            ins = pc[:, -2]
            
            sess, ops = get_model(model_path, batch_size=1, num_point=ps.shape[0])
            segp = inference(sess, ops, np.expand_dims(ps, 0), batch_size=1)
            segp = segp.squeeze()
            
            # Integrated
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum(seg==l)  # TP+FN
                total_pred_class[l] += np.sum(segp==l)  # TP+FP
                total_correct_class[l] += (np.sum((segp==l) & (seg==l)))  # TP
            
            # Stemwork segmentation F1 with respect to the main stem
            total_seen_stemwork_on_stem += np.sum((seg == 2) & (ins == 3))  # TP + FN
            total_pred_stemwork_on_stem += np.sum((segp == 2) & (ins == 3))  # TP + FP
            total_correct_stemwork_on_stem += np.sum((seg == 2) & (segp == 2) & (ins == 3))  # TP
            
            # Stemwork segmentation F1 with respect to individual morphological levels
            for rank in range(10):
                total_seen_stemwork_on_levels[rank] += np.sum((seg == 2) & (ins == rank + 6))  # TP + FN
                total_pred_stemwork_on_levels[rank] += np.sum((segp == 2) & (ins == rank + 6))  # TP + FP
                total_correct_stemwork_on_levels[rank] += np.sum((seg == 2) & (segp == 2) & (ins == rank + 6))  # TP
            
            # Stemwork segmentation F1 with respect to cultivars
            if (pc_name == "Harvest_01_PotNr_145") | (pc_name == "Harvest_02_PotNr_237"):
                total_seen_stemwork_cultivar[0] += np.sum(seg == 2)  # TP+FN
                total_pred_stemwork_cultivar[0] += np.sum(segp == 2)  # TP+FP
                total_correct_stemwork_cultivar[0] += np.sum((segp == 2) & (seg == 2))  # TP
            if (pc_name == "Harvest_02_PotNr_135") | (pc_name == "Harvest_01_PotNr_179"):
                total_seen_stemwork_cultivar[1] += np.sum(seg == 2)  # TP+FN
                total_pred_stemwork_cultivar[1] += np.sum(segp == 2)  # TP+FP
                total_correct_stemwork_cultivar[1] += np.sum((segp == 2) & (seg == 2))  # TP
            if pc_name == "Harvest_01_PotNr_80":
                total_seen_stemwork_cultivar[2] += np.sum(seg == 2)  # TP+FN
                total_pred_stemwork_cultivar[2] += np.sum(segp == 2)  # TP+FP
                total_correct_stemwork_cultivar[2] += np.sum((segp == 2) & (seg == 2))  # TP
            
            # Output Visualisation Files
            output_color_point_cloud(ps, seg, save_folder + '/gt_%d.obj' % (i), color_map)
            output_color_point_cloud(ps, segp, save_folder + '/pred_%d.obj' % (i), color_map)
            output_color_point_cloud(ps, segp == seg, save_folder + '/diff_%d.obj' % (i), lambda eq: (0, 1, 0) if eq else (1, 0, 0))
            
            idx = np.where(segp == 2)[0].tolist()
            pc_stemwork = ps[idx, 0:3]
            
            center = normalisation_info.iloc[i, 0:3].values
            scale = normalisation_info.iloc[i, 3]
            pc_restored_stemwork = pc_stemwork / scale + center
            
            np.savetxt(save_folder + "/pred_%d.csv" % (i), pc_restored_stemwork, delimiter=',')
        
        # F1 score (Integrated)
        for i in range(NUM_CLASSES):
            recall = total_correct_class[i] / total_seen_class[i]
            if total_pred_class[i] == 0:
                precision = 0
            else:
                precision = total_correct_class[i] / total_pred_class[i]
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1_rec[repeat_id, 0, i] = recall
            f1_rec[repeat_id, 1, i] = precision
            f1_rec[repeat_id, 2, i] = f1
        
        # F1 Score with respect to the Main Stem
        recall = total_correct_stemwork_on_stem / total_seen_stemwork_on_stem
        if total_pred_stemwork_on_stem == 0:
            precision = 0
        else:
            precision = total_correct_stemwork_on_stem / total_pred_stemwork_on_stem
        if precision + recall == 0:
            f1 = 0
        else:
            f1 =  2 * precision * recall / (precision + recall)
        f1_stemwork_on_stem_rec[repeat_id, 0] = f1
    
        # F1 score with respect to individual morphological levels
        for rank in range(10):
            recall = total_correct_stemwork_on_levels[rank] / total_seen_stemwork_on_levels[rank]
            if total_pred_stemwork_on_levels[rank] == 0:
                precision = 0
            else:
                precision = total_correct_stemwork_on_levels[rank] / total_pred_stemwork_on_levels[rank]
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1_stemwork_on_levels_rec[repeat_id, rank] = f1
    
        # F1 score with respect to cultivars
        for cultivar_id in range(3):
            recall = total_correct_stemwork_cultivar[cultivar_id] / total_seen_stemwork_cultivar[cultivar_id]
            if total_pred_stemwork_cultivar[cultivar_id] == 0:
                precision = 0
            else:
                precision = total_correct_stemwork_cultivar[cultivar_id] / total_pred_stemwork_cultivar[cultivar_id]
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            f1_stemwork_cultivar_rec[repeat_id, cultivar_id] = f1
        
        # IoU
        for i in range(NUM_CLASSES):
            iou = total_correct_class[i] / (total_seen_class[i] + total_pred_class[i] - total_correct_class[i])  # TP / (TP + FP + FN)
            iou_rec[repeat_id, i] = iou
    
    # Class-ave F1 score and IoU
    cls_ave_f1 = np.zeros((1, NUM_REPEATS), dtype="float32")
    cls_ave_iou = np.zeros((1, NUM_REPEATS), dtype="float32")
    for i in range(NUM_REPEATS):
        cls_ave_f1[0, i] = np.mean(f1_rec[i, 2,:])
        cls_ave_iou[0, i] = np.mean(iou_rec[i, :])
    
    # Record the performance into an Excel document
    eval_results = {"Ave Soil F1": [np.mean(f1_rec[:, 2, 0])], \
                    "Ave Soil IoU": [np.mean(iou_rec[:, 0])], \
                    "Ave Stick F1": [np.mean(f1_rec[:, 2, 1])], \
                    "Ave Stick IoU": [np.mean(iou_rec[:, 1])], \
                    "Ave Stemwork F1": [np.mean(f1_rec[:, 2, 2])], \
                    "Ave Stemwork IoU": [np.mean(iou_rec[:, 2])], \
                    "Ave Leaf F1": [np.mean(f1_rec[:, 2, 3])], \
                    "Ave Leafk IoU": [np.mean(iou_rec[:, 3])], \
                    "Ave Class-ave F1": [np.mean(cls_ave_f1)], \
                    "Ave Class-ave IoU": [np.mean(cls_ave_iou)], \
                    "Ave Stemwork F1 Merlice": [np.mean(f1_stemwork_cultivar_rec[:, 0])], \
                    "Ave Stemwork F1 Brioso": [np.mean(f1_stemwork_cultivar_rec[:, 1])], \
                    "Ave Stemwork F1 Gardener": [np.mean(f1_stemwork_cultivar_rec[:, 2])], \
                    "Ave Stemwork F1 on Stem": [np.mean(f1_stemwork_on_stem_rec)], \
                    "Ave Stemwork F1 on Level 01": [np.mean(f1_stemwork_on_levels_rec[:, 0])], \
                    "Ave Stemwork F1 on Level 02": [np.mean(f1_stemwork_on_levels_rec[:, 1])], \
                    "Ave Stemwork F1 on Level 03": [np.mean(f1_stemwork_on_levels_rec[:, 2])], \
                    "Ave Stemwork F1 on Level 04": [np.mean(f1_stemwork_on_levels_rec[:, 3])], \
                    "Ave Stemwork F1 on Level 05": [np.mean(f1_stemwork_on_levels_rec[:, 4])], \
                    "Ave Stemwork F1 on Level 06": [np.mean(f1_stemwork_on_levels_rec[:, 5])], \
                    "Ave Stemwork F1 on Level 07": [np.mean(f1_stemwork_on_levels_rec[:, 6])], \
                    "Ave Stemwork F1 on Level 08": [np.mean(f1_stemwork_on_levels_rec[:, 7])], \
                    "Ave Stemwork F1 on Level 09": [np.mean(f1_stemwork_on_levels_rec[:, 8])], \
                    "Ave Stemwork F1 on Level 10": [np.mean(f1_stemwork_on_levels_rec[:, 9])]}
    
    df = pd.DataFrame(eval_results)
    df.to_excel(OUTPUT_DIR + "/eval_results.xlsx", index=False, na_rep=0, inf_rep=0)
    
    print("Average stemwork F1 score: %f" % (np.mean(f1_rec[:, 2, 2])))
    # print("Average stemwork IoU: %f" % (np.mean(iou_rec[:, 2])))
    print('Average avg-class F1 score: %f' % (np.mean(cls_ave_f1)))
    # print('Average avg-class IoU: %f' % (np.mean(cls_ave_iou)))
    print('The best attempt with highest stemwork F1 score: ', np.argmax(f1_rec[:, 2, 2]))