'''
    Provider for dataset 
'''
import os
import os.path
import numpy as np
import time

class SceneflowDataset():
    def __init__(self, root='/tmp/FlyingThings3D_subset_processed_35m', npoints=8192, mode = 'train_ft3d'):
        self.npoints = npoints
        self.mode = mode
        self.root = root

        if self.mode == 'eval_kitti':

            self.samples = self.make_dataset()
            self.datapath = root        
            self.file_list = os.listdir(self.datapath)
            self.npoints = 16384

        elif self.mode == 'train_ft3d':
            self.datapath = os.path.join(self.root, 'train')
            self.file_list = os.listdir(self.datapath)
        
        elif self.mode == 'eval_ft3d':
            self.datapath = os.path.join(self.root, 'val')
            self.file_list = os.listdir(self.datapath)
        


    def __getitem__(self, index):

        np.random.seed(0)

        if self.mode == 'eval_kitti':
            fn = self.samples[index] 

        else:
            fn = self.file_list[index]
            fn = os.path.join(self.datapath, fn)
            
        pc1 = os.path.join(fn,'pc1.npy')
        pc2 = os.path.join(fn,'pc2.npy')

        with open(pc1, 'rb') as fp:
            pos1 = np.load(fp)

        with open(pc2, 'rb') as fp2:
            pos2 = np.load(fp2)
        
        flow = pos2[:, :3] - pos1[:, :3]
        
        if self.mode == 'eval_kitti':

            is_ground = np.logical_or(pos1[:,1] < -1.35, pos2[:,1] < -1.35)

            not_ground = np.logical_not(is_ground)

            near_mask = np.logical_and(pos1[:, 2] < 35, pos2[:, 2] < 35)
            near_mask = np.logical_and(not_ground, near_mask)

            indices = np.where(near_mask)[0]
        
        else:
            near_mask = np.logical_and(pos1[:, 2] < 35, pos2[:, 2] < 35)
            indices = np.where(near_mask)[0]

            
        if len(indices) >= self.npoints:
            sample_idx1 = np.random.choice(indices, self.npoints, replace=False)
        else:
            sample_idx1 = np.concatenate((indices, np.random.choice(indices, self.npoints - len(indices), replace=True)), axis=-1)
        
        if len(indices) >= self.npoints:
            sample_idx2 = np.random.choice(indices, self.npoints, replace=False)
        else:
            sample_idx2 = np.concatenate((indices, np.random.choice(indices, self.npoints - len(indices), replace=True)), axis=-1)

        pos1 = pos1[sample_idx1, :]
        pos2 = pos2[sample_idx2, :]
        flow = flow[sample_idx1, :]

        if self.mode == 'eval_kitti':
            return pos1, pos2, flow, fn
        else:
            return pos1, pos2, flow

    def __len__(self):
        return len(self.file_list)

    
    def make_dataset(self):
        
        do_mapping = True
        root = os.path.realpath(os.path.expanduser(self.root))

        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))

        if do_mapping:
            mapping_path = os.path.join(os.path.dirname(__file__), 'KITTI_mapping.txt')
            print('mapping_path', mapping_path)

            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(os.path.split(path)[-1])] != '']

        res_paths = useful_paths

        return res_paths




