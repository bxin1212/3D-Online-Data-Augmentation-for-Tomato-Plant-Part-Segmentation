import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold
import h5py
import copy


parser = argparse.ArgumentParser()


### - Parameters below should always be adjusted.
## - GPU & model selection
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: model]')

## - Dataset parameters
parser.add_argument('--dataset_path', default='./dataset/inner_point_removal/Pos_Norm_Color_Sem Ins_Test DS (45 pcs)/training_set.h5', help='Dataset path')
parser.add_argument('--num_classes', type=int, default=4, help='Number of semantic classes [default: 4]')
parser.add_argument('--num_point', type=int, default=50000, help='Point Number [default: 2048]')
parser.add_argument('--num_features', type=int, default=51, help='Demension of point features including positional info [default: 51]')

## - Parameters for training
parser.add_argument('--isCV', type=bool, default=False, help='Whether a cross validation process will be employed [default: False]')
parser.add_argument('--num_splits', type=int, default=7, help='Number of splits in cross validation [default: 7]')
parser.add_argument('--num_repeats', type=int, default=10, help='Number of training times for a single network configuration and dataset [default: 5]')
parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.005, help='Initial learning rate [default: 0.001]')
parser.add_argument('--pn_origin', type=bool, default=False, help='Whether to use the original number of S&G (feature propagation) layers in training [default: True]')

# - Parameters for focal loss
parser.add_argument('--isFocal_Loss', type=bool, default=False, help='Whether to use focal loss function [default: False]')
parser.add_argument('--class_balance_loss_type', type=int, default=1, help='Select the type of the class balance loss [default: 2]')
parser.add_argument('--gamma', type=float, default=1.0, help='Tough-instance factor [default: 2.0]')
parser.add_argument('--alpha_v1', type=float, default=[[1.], [1.], [1.], [1.]], help='class-balance factor for focal loss V1 [default: [1.], [5.6], [3.1], [1.4]]')
parser.add_argument('--alpha_v2', type=float, default=[1.], help='class-balance factor for focal loss V2 [default: 0.25]]')

# - Parameters for data augmentation
parser.add_argument('--aug_strategy', default=['leaf crossover', 'random down sampling', 'leaf_rotate_z'], help='Log dir [default: none]')

parser.add_argument('--jitter_sigma', type=float, default=0.11, help='Sigma to generate the random disturbance for jittering [default: 0.01]')
parser.add_argument('--jitter_clip', type=float, default=0.55, help='Clip for the random disturbance of jittering [default: 0.05]')

parser.add_argument('--scale_rate', type=float, default=[0.5, 1.5], help='Upper & lower boundary for the scaling rate [default: 0.8, 1.25]')

parser.add_argument('--global_shift_range', type=float, default=0.03, help='Shift range for global translation augmentation method [default: 0.1]')

parser.add_argument('--rotate_xy_sigma', type=float, default=np.pi/90, help='Sigma for rotation angle with respect to X or Y axis in rotate_xy augmentation method [default: pi/90]')

parser.add_argument('--max_cropping_ratio_xy', type=float, default=0.15, help='Maximum cropping ratio for x and y direction [default: 0.1]')
parser.add_argument('--max_cropping_ratio_z', type=float, default=0.15, help='Maximum cropping ratio for z direction [default: 0.1]')

parser.add_argument('--brightness_transform_range', type=float, default=[0.6, 3.5], help='Range for brightness transform [default: 0.7, 2.0]')

parser.add_argument('--leaf_shift_sigma', type=float, default=0.25, help='Shift sigma for leaf translation augmentation method [default: 0.05]')
parser.add_argument('--leaf_shift_clip', type=float, default=1.25, help='Shift clip for leaf translation augmentation method [default: 0.25]')

parser.add_argument('--leaf_rotate_z_range', type=float, default=1*np.pi/4, help='Range for leaf rotation augmentation method around z axis [default: 3*np.pi/4]')

parser.add_argument('--leaf_rotate_pca_range', type=float, default=70*np.pi/180, help='Range for leaf rotation augmentation method around PCA axis [default: pi/3]')

parser.add_argument('--n_max_crossover_leaves', type=int, default=4, help='Maximum number of leaves per plant to be used to crossover [default: 4]')

### - Parameters below can stay fixed in most time.
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

FLAGS = parser.parse_args()

EPOCH_CNT = 0

GPU_INDEX = FLAGS.gpu

NUM_CLASSES = FLAGS.num_classes  # Soil (class 0), stick (class 1), stemwork (class 2), other bio-structures (class 3)
NUM_POINT = FLAGS.num_point
NUM_FEATURES = FLAGS.num_features  # If NUM_FEATURES=51, [X, Y, Z, x_norm, y_norm, z_norm, colour (3*15)]

isCV = FLAGS.isCV
NUM_SPLITS = FLAGS.num_splits
NUM_REPEATS = FLAGS.num_repeats
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.learning_rate
PN_ORIGIN = FLAGS.pn_origin
ISFOCAL_LOSS = FLAGS.isFocal_Loss
CLASS_BALANCE_LOSS_TYPE = FLAGS.class_balance_loss_type
GAMMA = FLAGS.gamma
ALPHA_V1 = FLAGS.alpha_v1
ALPHA_V2 = FLAGS.alpha_v2
AUG_STRATEGY = FLAGS.aug_strategy
JITTER_SIGMA = FLAGS.jitter_sigma
JITTER_CLIP = FLAGS.jitter_clip
SCALE_RATE = FLAGS.scale_rate
GLOBAL_SHIFT_RANGE = FLAGS.global_shift_range
ROTATE_XY_SIGMA = FLAGS.rotate_xy_sigma
MAX_CROPPING_RATIO_XY = FLAGS.max_cropping_ratio_xy
MAX_CROPPING_RATIO_Z = FLAGS.max_cropping_ratio_z
BRIGHTNESS_TRANSFORM_RANGE = FLAGS.brightness_transform_range
LEAF_SHIFT_SIGMA = FLAGS.leaf_shift_sigma
LEAF_SHIFT_CLIP = FLAGS.leaf_shift_clip
LEAF_ROTATE_Z_RANGE = FLAGS.leaf_rotate_z_range
LEAF_ROTATE_PCA_RANGE = FLAGS.leaf_rotate_pca_range
N_MAX_CROSSOVER_LEAVES = FLAGS.n_max_crossover_leaves

MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

LOG_DIR = os.path.join(ROOT_DIR, FLAGS.log_dir)
my_file = Path(LOG_DIR)
if not my_file.is_dir(): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# Get dataset
DATASET_PATH = FLAGS.dataset_path
all_data = h5py.File(DATASET_PATH, "r")

# Get additional training point cloud information
if ("leaf crossover" in AUG_STRATEGY) | ("leaf_rotate_z" in AUG_STRATEGY) | ("leaf_rotate_pca" in AUG_STRATEGY) | ("leaf_translation" in AUG_STRATEGY):
    df_effective_leaf_num = pd.read_excel(os.path.join(os.path.dirname(DATASET_PATH), "training_set_effective_leaf_number.xlsx"), \
                                          index_col=0)
    df_train_set_leaf_info = pd.read_excel(os.path.join(os.path.dirname(DATASET_PATH), "training_set_leaf_info.xlsx"), \
                                           index_col=0)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
    

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate 
       

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def get_network_config():

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FEATURES)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            if PN_ORIGIN:
                pred, end_points = MODEL.get_model_origin(pointclouds_pl, \
                                       is_training_pl, NUM_FEATURES, NUM_CLASSES, bn_decay=bn_decay)
            else:
                pred, end_points = MODEL.get_model_simplify(pointclouds_pl, \
                                       is_training_pl, NUM_FEATURES, NUM_CLASSES, bn_decay=bn_decay)
            if ISFOCAL_LOSS:
                if CLASS_BALANCE_LOSS_TYPE == 1:
                    loss, debug_port = MODEL.get_focal_loss_v1(pred, labels_pl, GAMMA, ALPHA_V1, NUM_CLASSES)
                else:
                    loss, debug_port = MODEL.get_focal_loss_v2(pred, labels_pl, GAMMA, ALPHA_V2, NUM_CLASSES)
            else:
                loss, debug_port = MODEL.get_loss(pred, labels_pl, NUM_CLASSES)
            tf.summary.scalar('loss', loss)
            
            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        if isCV:
            validation_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'validation'), sess.graph)
        else:
            validation_writer = None

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'debug_port': debug_port,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}
    
    return sess, \
           ops, \
           saver, \
           train_writer, \
           validation_writer
           

def train_with_cross_validation():
    global EPOCH_CNT
    
    kf = KFold(n_splits=NUM_SPLITS, shuffle=True)
    
    fold_nr = 0
    best_class_ave_f1_rec = np.zeros((1, NUM_SPLITS), dtype=np.float32)
    best_stemwork_f1_rec = np.zeros((1, NUM_SPLITS), dtype=np.float32)
    for train_id, val_id in kf.split(all_data):
    
        EPOCH_CNT = 0
        
        cv_save_folder = os.path.join(LOG_DIR, "val_" + str(fold_nr))
        my_file = Path(cv_save_folder)
        if not my_file.is_dir(): os.mkdir(cv_save_folder)
        
        all_file_list = np.array(list(all_data.keys()))
        train_file_list = all_file_list[train_id]
        val_file_list = all_file_list[val_id]
        
        sess, \
        ops, \
        saver, \
        train_writer, \
        validation_writer = get_network_config()
        
        train_loss_rec = np.zeros((1, MAX_EPOCH), "float32")
        train_acc_rec = np.zeros((1, MAX_EPOCH), "float32")
        train_individual_class_rec = np.zeros((NUM_CLASSES, 3, MAX_EPOCH), "float32")
        eval_loss_rec = np.zeros((1, MAX_EPOCH), "float32")
        eval_overall_acc_rec = np.zeros((1, MAX_EPOCH), "float32")
        eval_individual_class_rec = np.zeros((NUM_CLASSES, 3, MAX_EPOCH), "float32")
        
        best_f1_target_class = 0
        for epoch in range(MAX_EPOCH):
            sys.stdout.flush()
             
            train_loss, \
            train_acc, \
            train_individual_class = train_one_epoch(sess=sess, \
                                                     ops=ops, \
                                                     train_writer=train_writer, \
                                                     train_file_list=train_file_list, \
                                                     additional_input=fold_nr)
            eval_loss, \
            eval_overall_acc, \
            eval_individual_class = eval_one_epoch(sess=sess, \
                                                   ops=ops, \
                                                   validation_writer=validation_writer, \
                                                   val_file_list=val_file_list, \
                                                   fold_nr=fold_nr)
            
            train_loss_rec[0, epoch] = train_loss
            train_acc_rec[0, epoch] = train_acc
            eval_loss_rec[0, epoch] = eval_loss
            eval_overall_acc_rec[0, epoch] = eval_overall_acc
            for class_id in range(NUM_CLASSES):
                eval_individual_class_rec[class_id, :, epoch] = eval_individual_class[:, class_id]
            for class_id in range(NUM_CLASSES):
                train_individual_class_rec[class_id, :, epoch] = train_individual_class[:, class_id]
            
            # Save the model with the best eval target-class F1 score to the local disk.
            if np.mean(eval_individual_class[2,2]) > best_f1_target_class:
                save_path = saver.save(sess, os.path.join(cv_save_folder, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
                best_f1_target_class = np.mean(eval_individual_class[2, 2])
                
            EPOCH_CNT += 1
                
        class_ave_f1 = np.mean(eval_individual_class_rec[:, 2, :], axis=0)
        
        np.savetxt(cv_save_folder + "/train_loss_rec.csv", train_loss_rec)
        np.savetxt(cv_save_folder + "/train_acc_rec.csv", train_acc_rec)
        np.savetxt(cv_save_folder + "/eval_loss_rec.csv", eval_loss_rec)
        np.savetxt(cv_save_folder + "/eval_overall_acc_rec.csv", eval_overall_acc_rec)
        for class_id in range(NUM_CLASSES):
            np.savetxt(cv_save_folder + "/train_rec_class_" + str(class_id) + ".csv", \
                       train_individual_class_rec[class_id])
        for class_id in range(NUM_CLASSES):
            np.savetxt(cv_save_folder + "/eval_rec_class_" + str(class_id) + ".csv", \
                       eval_individual_class_rec[class_id])
        
        best_class_ave_f1_this = np.max(class_ave_f1)
        best_stemwork_f1_this = np.max(eval_individual_class_rec[2, 2, :])
        best_class_ave_f1_rec[0, fold_nr] = best_class_ave_f1_this
        best_stemwork_f1_rec[0, fold_nr] = best_stemwork_f1_this
        
        fold_nr += 1
        
    np.savetxt("./log/best_class_ave_f1_rec.csv", best_class_ave_f1_rec)
    np.savetxt("./log/best_stemwork_f1_rec.csv", best_stemwork_f1_rec)
        
    log_string("Ave best class-average f1 score: " + str(np.mean(best_class_ave_f1_rec)))
    log_string("Ave best stemwork f1 score: " + str(np.mean(best_stemwork_f1_rec)))


def train_without_cross_validation():
    global EPOCH_CNT
    
    for repeat_id in range(NUM_REPEATS):
        
        EPOCH_CNT = 0
        
        save_folder = os.path.join(LOG_DIR, "repeat_" + str(repeat_id))
        my_file = Path(save_folder)
        if not my_file.is_dir(): os.mkdir(save_folder)
        
        train_file_list = np.array(list(all_data.keys()))
        
        sess, \
        ops, \
        saver, \
        train_writer, \
        _ = get_network_config()
        
        train_loss_rec = np.zeros((1, MAX_EPOCH), "float32")
        train_acc_rec = np.zeros((1, MAX_EPOCH), "float32")
        train_individual_class_rec = np.zeros((NUM_CLASSES, 3, MAX_EPOCH), "float32")
        
        for epoch in range(MAX_EPOCH):
            sys.stdout.flush()
             
            train_loss, \
            train_acc, \
            train_individual_class = train_one_epoch(sess=sess, \
                                                     ops=ops, \
                                                     train_writer=train_writer, \
                                                     train_file_list=train_file_list, \
                                                     additional_input=repeat_id)
        
            train_loss_rec[0, epoch] = train_loss
            train_acc_rec[0, epoch] = train_acc
            for class_id in range(NUM_CLASSES):
                train_individual_class_rec[class_id, :, epoch] = train_individual_class[:, class_id]
            
            # Save the models to the local disk every ten epoches.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(save_folder, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
            
            EPOCH_CNT += 1
    
        np.savetxt(save_folder + "/train_loss_rec.csv", train_loss_rec)
        np.savetxt(save_folder + "/train_acc_rec.csv", train_acc_rec)
        for class_id in range(NUM_CLASSES):
            np.savetxt(save_folder + "/train_rec_class_" + str(class_id) + ".csv", \
                       train_individual_class_rec[class_id])


def get_batch(file_list, idxs, start_idx, end_idx, is_training=True):
    
    bsize = end_idx-start_idx  # Batch size
    batch_data = np.zeros((bsize, NUM_POINT, NUM_FEATURES))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    
    ## - Data augmentation
    # - Global augmentation
    if "cropping" in AUG_STRATEGY:
        all_raw_batch_pcs = list()
        for i in range(bsize):
            pc_file_name = file_list[idxs[i+start_idx]]
            pc = all_data.get(pc_file_name)[()]
            pc = provider.cropping(pc, 
                                   max_cropping_ratio_xy=MAX_CROPPING_RATIO_XY, \
                                   max_cropping_ratio_z=MAX_CROPPING_RATIO_Z)
            all_raw_batch_pcs.append(pc)
    # - Local augmentation
    if "leaf crossover" in AUG_STRATEGY:
        all_raw_batch_pcs = list()
        batch_pc_file_name = list()
        for i in range(bsize):
            pc_file_name = file_list[idxs[i+start_idx]]
            pc = all_data.get(pc_file_name)[()]
            all_raw_batch_pcs.append(pc)
            batch_pc_file_name.append(pc_file_name)
        
        batch_cultivars = df_effective_leaf_num.loc["Cultivars", batch_pc_file_name]
        batch_leaf_numbers = df_effective_leaf_num.loc["Number of effective leaves", batch_pc_file_name]
        batch_start_rank = df_effective_leaf_num.loc["Start rank", batch_pc_file_name]
        batch_leaf_base = df_train_set_leaf_info[batch_pc_file_name]
        
        if sum(batch_cultivars == "Merlice") > 1:
            cultivar = "Merlice"
            all_raw_batch_pcs = provider.leaf_crossover(all_raw_batch_pcs, batch_pc_file_name, \
                                    batch_cultivars, batch_leaf_numbers, batch_start_rank, batch_leaf_base, \
                                    cultivar=cultivar, num_features=NUM_FEATURES, n_max_crossover_leaves=N_MAX_CROSSOVER_LEAVES)
        if sum(batch_cultivars == "Brioso") > 1:
            cultivar = "Brioso"
            all_raw_batch_pcs = provider.leaf_crossover(all_raw_batch_pcs, batch_pc_file_name, \
                                    batch_cultivars, batch_leaf_numbers, batch_start_rank, batch_leaf_base, \
                                    cultivar=cultivar, num_features=NUM_FEATURES, n_max_crossover_leaves=N_MAX_CROSSOVER_LEAVES)
        if sum(batch_cultivars == "GardenersDelight") > 1:
            cultivar = "GardenersDelight"
            all_raw_batch_pcs = provider.leaf_crossover(all_raw_batch_pcs, batch_pc_file_name, \
                                    batch_cultivars, batch_leaf_numbers, batch_start_rank, batch_leaf_base, \
                                    cultivar=cultivar, num_features=NUM_FEATURES, n_max_crossover_leaves=N_MAX_CROSSOVER_LEAVES)
        
    for i in range(bsize):
        
        if ("leaf crossover" in AUG_STRATEGY) | ("cropping" in AUG_STRATEGY):
            pc = all_raw_batch_pcs[i]
        else:
            pc_file_name = file_list[idxs[i+start_idx]]
            pc = all_data.get(pc_file_name)[()]
        
        ## - Data augmentation
        # - Global augmentation
        if "random down sampling" in AUG_STRATEGY:
            pc = provider.random_down_sampling(pc, NUM_POINT)
        # - Local augmentation
        if "leaf_translation" in AUG_STRATEGY:
            pc_pos = copy.deepcopy(pc[:, 0:3])
            instance_label = pc[:, -2]
            start_rank = df_effective_leaf_num.loc["Start rank", pc_file_name]
            n_leaves = df_effective_leaf_num.loc["Number of effective leaves", pc_file_name]
            pc[:, 0:3] = provider.leaf_shift(pc_pos, instance_label, start_rank, n_leaves, \
                                             sigma=LEAF_SHIFT_SIGMA, clip=LEAF_SHIFT_CLIP)
        if "leaf_rotate_z" in AUG_STRATEGY:
            pc_pos = copy.deepcopy(pc[:, 0:3])
            instance_label = pc[:, -2]
            start_rank = df_effective_leaf_num.loc["Start rank", pc_file_name]
            n_leaves = df_effective_leaf_num.loc["Number of effective leaves", pc_file_name]
            leaf_base = df_train_set_leaf_info[pc_file_name]
            pc[:, 0:3] = provider.leaf_rotate_z(pc_pos, instance_label, start_rank, leaf_base, n_leaves, \
                                                rotate_range=LEAF_ROTATE_Z_RANGE)
        
        if "leaf_rotate_pca" in AUG_STRATEGY:
            pc_pos = copy.deepcopy(pc[:, 0:3])
            instance_label = pc[:, -2]
            start_rank = df_effective_leaf_num.loc["Start rank", pc_file_name]
            n_leaves = df_effective_leaf_num.loc["Number of effective leaves", pc_file_name]
            leaf_base = df_train_set_leaf_info[pc_file_name]
            pc[:, 0:3] = provider.leaf_rotate_pca(pc_pos, instance_label, start_rank, leaf_base, n_leaves, \
                                                rotate_range=LEAF_ROTATE_PCA_RANGE)
        
        ps = pc[:, 0:3]
        feature = pc[:, 3:NUM_FEATURES]
        semantic_seg = pc[:, -1]
        batch_data[i,:,0:3] = ps
        batch_data[i,:,3:NUM_FEATURES] = feature
        batch_label[i,:] = semantic_seg
        
    ## - Other data augmentation methods
    # - Global augmentation
    if "jittering" in AUG_STRATEGY:
        pc_pos = copy.deepcopy(batch_data[:, :, 0:3])
        batch_data[:, :, 0:3] = provider.jitter_point_cloud(pc_pos, sigma=JITTER_SIGMA, clip=JITTER_CLIP)
    if "scaling_xy" in AUG_STRATEGY:
        pc_pos = copy.deepcopy(batch_data[:, :, 0:3])
        batch_data[:, :, 0:3] = provider.random_scale_point_cloud_xy(pc_pos, scale_low=SCALE_RATE[0], scale_high=SCALE_RATE[1])
    if "translation" in AUG_STRATEGY:
        pc_pos = copy.deepcopy(batch_data[:, :, 0:3])
        batch_data[:, :, 0:3] = provider.shift_point_cloud(pc_pos, shift_range=GLOBAL_SHIFT_RANGE)
    if "rotate_z" in AUG_STRATEGY:
        pc_pos = copy.deepcopy(batch_data[:, :, 0:3])
        batch_data[:, :, 0:3] = provider.rotate_point_cloud_z(pc_pos)
    if "rotate_xy" in AUG_STRATEGY:
        pc_pos = copy.deepcopy(batch_data[:, :, 0:3])
        batch_data[:, :, 0:3] = provider.rotate_point_cloud_xy(pc_pos, sigma=ROTATE_XY_SIGMA)
    if "brightness_transform" in AUG_STRATEGY:
        pc_col = copy.deepcopy(batch_data[:, :, 6:NUM_FEATURES])
        batch_data[:, :, 6:NUM_FEATURES] = provider.brightness_transform(pc_col, brightness_transform_range=BRIGHTNESS_TRANSFORM_RANGE)
    
    return batch_data, batch_label


def train_one_epoch(sess, ops, train_writer, train_file_list, additional_input):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(train_file_list))
    np.random.shuffle(train_idxs)
    num_batches = len(train_file_list)/BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_pred_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    for batch_idx in range(int(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label = get_batch(train_file_list, train_idxs, start_idx, end_idx, is_training=is_training)
        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, _, loss_val, pred_val, debug_port = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred'], ops['debug_port']], feed_dict=feed_dict)
        
        # - Debug flag visualisation -
        # print(debug_port)
        # ----------------------------

        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label==l)  # TP+FN
            total_pred_class[l] += np.sum(pred_val==l)  # TP+FP
            total_correct_class[l] += (np.sum((pred_val==l) & (batch_label==l)))  # TP
    
    train_loss = loss_sum
    train_acc = total_correct / float(total_seen)
    
    eval_individual_class = np.zeros((3, NUM_CLASSES), dtype="float32")
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
        eval_individual_class[0, i] = recall
        eval_individual_class[1, i] = precision
        eval_individual_class[2, i] = f1
    
    if isCV:
        log_string('---- Fold No.%d ---- EPOCH %03d TRAINING ----'%(additional_input, EPOCH_CNT))
    else:
        log_string('---- Repeat No.%d ---- EPOCH %03d TRAINING ----'%(additional_input, EPOCH_CNT))
    
    log_string('mean loss: %f' % (loss_sum / num_batches))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
    log_string('train avg class F1 score: %f' % (np.mean(eval_individual_class[2, :])))
    
    return train_loss, \
           train_acc, \
           eval_individual_class
        

def eval_one_epoch(sess, ops, validation_writer, val_file_list, fold_nr):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(val_file_list))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = np.ceil(len(val_file_list) / BATCH_SIZE)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_pred_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('---- Fold No.%d ---- EPOCH %03d EVALUATION ----'%(fold_nr, EPOCH_CNT))
    
    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, NUM_FEATURES))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)
    for batch_idx in range(int(num_batches)):
        if batch_idx %20==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(val_file_list), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        cur_batch_data, cur_batch_label = get_batch(val_file_list, test_idxs, start_idx, end_idx, is_training=is_training)
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label

        # ---------------------------------------------------------------------
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        validation_writer.add_summary(summary, step)
        # ---------------------------------------------------------------------
        
        # Select valid data
        cur_pred_val = pred_val[0:cur_batch_size]
        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        for i in range(cur_batch_size):
            logits = cur_pred_val_logits[i,:,:]
            cur_pred_val[i,:] = np.argmax(logits, 1).T
        correct = np.sum(cur_pred_val == cur_batch_label)
        
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT)
        loss_sum += loss_val
        
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(cur_batch_label==l)  # TP+FN
            total_pred_class[l] += np.sum(cur_pred_val==l)  # TP+FP
            total_correct_class[l] += (np.sum((cur_pred_val==l) & (cur_batch_label==l)))  # TP
    
    eval_loss = loss_sum / float(len(val_file_list)/BATCH_SIZE)
    eval_overall_acc = total_correct / float(total_seen)
    
    eval_individual_class = np.zeros((3, NUM_CLASSES), dtype="float32")
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
        eval_individual_class[0, i] = recall
        eval_individual_class[1, i] = precision
        eval_individual_class[2, i] = f1
        log_string('eval class ' + str(i) + ' F1 score: %f' % (f1))
    log_string('eval avg class F1 score: %f' % (np.mean(eval_individual_class[2,:])))
    
    return eval_loss, \
           eval_overall_acc, \
           eval_individual_class


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    
    if isCV:
        train_with_cross_validation()
    else:
        train_without_cross_validation()
    
    all_data.close()
    
    LOG_FOUT.close()
