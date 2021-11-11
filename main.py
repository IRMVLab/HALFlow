'''
    Single-GPU training code
'''

import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import dataset
import pickle

from main_util import get_2d_flow, scene_flow_EPE_np, evaluate_2d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type = str, default='train', help=' running mode: train / eval ')
parser.add_argument('--dataset', type = str, default='ft3d', help=' data mode: ft3d / kitti')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='HALFlowNet', help='Model name ')
parser.add_argument('--data_ft3d_path', default='/tmp/FlyingThings3D_subset_processed_35m', help='FlytingThings3d Dataset directory')
parser.add_argument('--data_kitti_path', default='/tmp/KITTI_processed_occ_final', help='KITTI Dataset directory')
parser.add_argument('--log_dir', default='log_train', help='Log dir [default: log_train]')
parser.add_argument('--checkpoint_path', default='None', help='Checkpoint_path [default: None]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=1510, help='Epoch to run [default: 1510]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')##########decay############3
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

EPOCH_CNT = 0

MODE = FLAGS.mode
DATA_MODE = FLAGS.dataset

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DATA = FLAGS.data_ft3d_path
DATA_kitti = FLAGS.data_kitti_path
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
CHECKPOINT_PATH = FLAGS.checkpoint_path

MODEL = importlib.import_module(FLAGS.model) # import network module

MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
UTIL_FILE = os.path.join(BASE_DIR, 'utils/HALFlowNet_util.py')

LOG_DIR = FLAGS.log_dir + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (UTIL_FILE, LOG_DIR)) ###SAVE THE UTIL FILE
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
os.system('cp %s %s' % ('dataset.py', LOG_DIR)) # bkp of dataset file

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.7
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

TRAIN_DATASET = dataset.SceneflowDataset(DATA, npoints=NUM_POINT, mode = 'train_ft3d')
TEST_DATASET = dataset.SceneflowDataset(DATA, npoints=NUM_POINT, mode = 'eval_ft3d')
TEST_DATASET_kitti = dataset.SceneflowDataset(DATA_kitti, npoints=NUM_POINT, mode = 'eval_kitti')


train_dataset_idxs = np.arange(0, 19640)
np.random.shuffle(train_dataset_idxs)
train_dataset_idxs_quarter = train_dataset_idxs[:4910]


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

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):

            pointclouds_pl, labels_pl, pointclouds_pl_raw = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0.0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            l0_pred, l1_pred, l2_pred, l3_pred, l0_label, l1_label, l2_label, l3_label, pc1_sample, pc2_sample = MODEL.get_model(pointclouds_pl, pointclouds_pl_raw, labels_pl, is_training_pl, dataset = DATA_MODE, bn_decay=bn_decay)
        
            loss = MODEL.get_loss(l0_pred, l1_pred, l2_pred, l3_pred, l0_label, l1_label, l2_label, l3_label)
        
            tf.summary.scalar('loss', loss)

            print("--- Get training operator")
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
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
        
        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_pl_raw': pointclouds_pl_raw,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': l0_pred,
               'label_2048': l0_label,
               'pc1': pc1_sample,
               'pc2': pc2_sample,               
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               }
               
        if MODE == 'train':    

            if CHECKPOINT_PATH != None:
                saver.restore(sess, CHECKPOINT_PATH)
                log_string ("model restored")

            else:
                # Init variables
                init = tf.global_variables_initializer()
                sess.run(init)


            epe3d_min = 10000.0

            for epoch in range(0, MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()
                
                train_one_epoch(sess, ops, train_writer)

                epe3d, acc3d_1, acc3d_2, outlier = eval_one_epoch(sess, ops, test_writer)

                # Save the variables to disk.
                if epe3d < epe3d_min:
                    epe3d_min = epe3d
                    save_path_epe3d = saver.save(sess, os.path.join(LOG_DIR, "model_ft3d_epe3d_%03d.ckpt" % (epoch)))
                    log_string("Model saved in file: %s" % save_path_epe3d)
        
        elif MODE == 'eval':

            saver.restore(sess, CHECKPOINT_PATH)
            log_string ("model restored")
            
            eval_one_epoch(sess, ops, test_writer)
            eval_one_epoch_kitti(sess, ops, test_writer)



def get_batch(dataset, idxs, start_idx, end_idx, mode = 'train_ft3d'):

    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT*2, 6))
    batch_data_raw = np.zeros((bsize, NUM_POINT*2, 6))
    batch_label = np.zeros((bsize, NUM_POINT, 3))
    paths = []

    shuffle_idx = np.arange(NUM_POINT)
    np.random.shuffle(shuffle_idx)

    for i in range(bsize):

        if mode == 'eval_kitti':
            pc1, pc2, flow, path = dataset[idxs[i+start_idx]]#####################################
            paths.append(path)
        else:
            pc1, pc2, flow= dataset[idxs[i+start_idx]]

        batch_data_raw[i,:NUM_POINT,:3] = pc1[shuffle_idx]
        batch_data_raw[i,NUM_POINT:,:3] = pc2[shuffle_idx]######get the raw xyz coordinates

        # move pc1 to center
        pc1_center = np.mean(pc1, 0)
        pc1 -= pc1_center
        pc2 -= pc1_center

        batch_data[i,:NUM_POINT,:3] = pc1[shuffle_idx]
        batch_data[i,NUM_POINT:,:3] = pc2[shuffle_idx]
        batch_label[i] = flow[shuffle_idx]
    
    if mode == 'eval_kitti':
        return batch_data, batch_data_raw, batch_label, paths
    else:
        return batch_data, batch_data_raw, batch_label



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    global EPOCH_CNT

    train_idxs = train_dataset_idxs_quarter

    np.random.shuffle(train_idxs)
    num_batches = 4910// BATCH_SIZE

    log_string(str(datetime.now()))

    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_data_raw, batch_label = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['pointclouds_pl_raw']: batch_data_raw,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}

        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_sum += loss_val

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            loss_sum = 0

    EPOCH_CNT += 1


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))

    num_batches = (len(TEST_DATASET)+BATCH_SIZE-1) // BATCH_SIZE

    loss_sum = 0
    loss_sum_l2 = 0
    sum_epe3d = 0
    sum_epe2d = 0
    sum_acc3d_1 = 0
    sum_acc3d_2 = 0
    sum_acc2d = 0
    sum_outlier = 0

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT*2, 3))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    batch_data_raw = np.zeros((BATCH_SIZE, NUM_POINT*2, 3))

    for batch_idx in range(num_batches):

        if batch_idx %20==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        cur_batch_data, cur_batch_data_raw, cur_batch_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx, mode = 'eval_ft3d')

        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
            batch_data_raw = cur_batch_data_raw
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label
            batch_data_raw[0:cur_batch_size] = cur_batch_data_raw

        # ---------------------------------------------------------------------
        # ---- INFERENCE BELOW ----

        feed_dict = {ops['pointclouds_pl']: batch_data,
                    ops['pointclouds_pl_raw']: batch_data_raw,
                    ops['labels_pl']: batch_label,
                    ops['is_training_pl']: is_training}

        loss_val, label_2048, pred_val, pc1, pc2 = sess.run([ops['loss'], ops['label_2048'],ops['pred'], ops['pc1'], ops['pc2']], feed_dict=feed_dict)

        # ---- INFERENCE ABOVE ----

        pc1 = pc1[:cur_batch_size, :, :]
        pred_val = pred_val[:cur_batch_size, :, :]
        label_2048 = label_2048[:cur_batch_size, :, :]

        flow_pred_2d, flow_gt_2d = get_2d_flow(pc1, pc1+label_2048, pc1+pred_val)
        tmp = np.sum((pred_val - label_2048)**2, 2) / 2.0

        loss_val_np = np.mean(tmp)
        loss_val = loss_val_np
        print('batch loss: %f' % (loss_val))
        
        loss_sum += loss_val
        EPE3D, acc3d_1, acc3d_2, outlier = scene_flow_EPE_np(pred_val, label_2048)
        EPE2D, acc2d = evaluate_2d(flow_pred_2d, flow_gt_2d)

        print('batch EPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f\tbatch EPE 2D: %f\tACC 2D: %f' % (EPE3D, acc3d_1, acc3d_2, EPE2D, acc2d))

        loss_sum += loss_val
        loss_sum_l2 += loss_val_np
        sum_epe3d += EPE3D
        sum_epe2d += EPE2D
        sum_acc3d_1 += acc3d_1
        sum_acc3d_2 += acc3d_2
        sum_acc2d += acc2d
        sum_outlier += outlier

        # Dump some results
        if batch_idx == 0:
            with open('test_results.pkl', 'wb') as fp:
                pickle.dump([batch_data, label_2048, pred_val], fp)
                
    epe3d = sum_epe3d / num_batches
    epe2d = sum_epe2d / num_batches
    acc2d = sum_acc2d / num_batches
    acc3d_1 = sum_acc3d_1 / num_batches
    acc3d_2 = sum_acc3d_2 / num_batches
    outlier = sum_outlier / num_batches######

    log_string('eval mean EPE 3D: %f' % (epe3d))
    log_string('eval mean EPE 2D: %f' % (epe2d))
    log_string('eval mean acc3d_1: %f' % (acc3d_1))
    log_string('eval mean acc3d_2 : %f' % (acc3d_2))
    log_string('eval mean acc2d : %f' % (acc2d))
    log_string('eval mean outlier : %f' % (outlier))
    log_string('eval mean loss: %f' % (loss_sum / num_batches))

    return epe3d, acc3d_1, acc3d_2, outlier


def eval_one_epoch_kitti(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT

    is_training = False
    test_idxs = np.arange(0, 142)

    num_batches = (142+BATCH_SIZE-1) // BATCH_SIZE

    loss_sum = 0
    loss_sum_l2 = 0
    sum_epe3d = 0
    sum_epe2d = 0
    sum_acc3d_1 = 0
    sum_acc3d_2 = 0
    sum_acc2d = 0
    sum_outlier = 0

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION_KITTI____ ----'%(EPOCH_CNT))

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT*2, 3))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    batch_data_raw = np.zeros((BATCH_SIZE, NUM_POINT*2, 3))

    for batch_idx in range(num_batches):
        
        if batch_idx %20==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE

        end_idx = min(142, (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        cur_batch_data, cur_batch_data_raw, cur_batch_label, paths = get_batch(TEST_DATASET_kitti, test_idxs, start_idx, end_idx, mode = 'eval_kitti')

        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
            batch_data_raw = cur_batch_data_raw
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label
            batch_data_raw[0:cur_batch_size] = cur_batch_data_raw

        # ---------------------------------------------------------------------
        # ---- INFERENCE BELOW ----

        feed_dict = {ops['pointclouds_pl']: batch_data,
                    ops['pointclouds_pl_raw']: batch_data_raw,
                    ops['labels_pl']: batch_label,
                    ops['is_training_pl']: is_training}

        loss_val, label_2048, pred_val, pc1, pc2 = sess.run([ops['loss'], ops['label_2048'],ops['pred'], ops['pc1'], ops['pc2']], feed_dict=feed_dict)

        pc1 = pc1[:cur_batch_size, :, :]
        pred_val = pred_val[:cur_batch_size, :, :]
        label_2048 = label_2048[:cur_batch_size, :, :]

        # ---- INFERENCE ABOVE ----

        flow_pred_2d, flow_gt_2d = get_2d_flow(pc1, pc1+label_2048, pc1+pred_val, paths)

        tmp = np.sum((pred_val - label_2048)**2, 2) / 2.0

        loss_val_np = np.mean(tmp)

        loss_val = loss_val_np
        print('batch loss: %f' % (loss_val))
        loss_sum += loss_val

        EPE3D, acc3d_1, acc3d_2, outlier = scene_flow_EPE_np(pred_val, label_2048)
        EPE2D, acc2d = evaluate_2d(flow_pred_2d, flow_gt_2d)

        print('batch EPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f\tEPE 2d 2: %f\tACC 2D: %f' % (EPE3D, acc3d_1, acc3d_2, EPE2D, acc2d))

        loss_sum += loss_val
        loss_sum_l2 += loss_val_np
        sum_epe3d += EPE3D
        sum_epe2d += EPE2D
        sum_acc3d_1 += acc3d_1
        sum_acc3d_2 += acc3d_2
        sum_acc2d += acc2d
        sum_outlier += outlier

        # Dump some results
        if batch_idx == 0:
            with open('test_results.pkl', 'wb') as fp:
                pickle.dump([batch_data, label_2048, pred_val], fp)
                
    epe3d = sum_epe3d / num_batches
    epe2d = sum_epe2d / num_batches
    acc2d = sum_acc2d / num_batches
    acc3d_1 = sum_acc3d_1 / num_batches
    acc3d_2 = sum_acc3d_2 / num_batches
    outlier = sum_outlier / num_batches


    log_string('KITTI eval mean EPE 3D: %f' % (epe3d))
    log_string('KITTI eval mean EPE 2D: %f' % (epe2d))
    log_string('KITTI eval mean acc3d_1: %f' % (acc3d_1))
    log_string('KITTI eval mean acc3d_2 : %f' % (acc3d_2))
    log_string('KITTI eval mean acc2d : %f' % (acc2d))
    log_string('KITTI eval mean outlier : %f' % (outlier))
    log_string('KITTI eval mean loss: %f' % (loss_sum / num_batches))

    return epe3d, acc3d_1, acc3d_2, outlier



if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
