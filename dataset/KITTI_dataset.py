'''
    Provider for duck dataset from xingyu liu
'''

import os
import os.path
import json
import numpy as np
import sys
import pickle
import glob


class SceneflowDataset():
    def __init__(self, root='data_preprocessing/data_processed_maxcut_35_both_mask_20k_2k', npoints=2048, train=True):
        self.npoints = npoints
        self.train = train
        self.root = root

        if self.train == 0:
            self.datapath = os.path.join(self.root, 'train')
            self.file_list = os.listdir(self.datapath)
        

        elif self.train == 1:

            self.datapath = os.path.join(self.root, 'val')
            self.file_list = os.listdir(self.datapath)
        

        elif self.train == 2:

            self.samples = self.make_dataset()################################################################
            self.datapath = root    #####################################KITTI######################     
            self.file_list = os.listdir(self.datapath)

    def __getitem__(self, index):

        if self.train ==2:
            fn = self.samples[index] 

        else:
            fn = self.file_list[index]
            fn = os.path.join(self.datapath, fn)
            
        print(fn)
        
        if fn == '../../tmp/train/0016130':
            fn = '../../tmp/train/0016131'

        if fn == '../../tmp/train/0021790':
            fn = '../../tmp/train/0016131'
        
        if fn == '../../tmp/train/0012550':
            fn = '../../tmp/train/0016131'
        
        if fn == '../../tmp/train/0004781':
            fn = '../../tmp/train/0016131'

        if fn == '../../tmp/train/0004641':
            fn = '../../tmp/train/0016131'


        if fn == 'KITTI_processed_occ_final/000080':
            fn = 'KITTI_processed_occ_final/000051'

        if fn == 'KITTI_processed_occ_final/000148':
            fn = 'KITTI_processed_occ_final/000051'

        if fn == 'FlyingThings3D_subset_processed_35m/train/0004641' or fn == 'FlyingThings3D_subset_processed_35m/train/0016130' or fn == 'FlyingThings3D_subset_processed_35m/train/0021790'or fn == 'FlyingThings3D_subset_processed_35m/train/0012550' or fn == 'FlyingThings3D_subset_processed_35m/train/0004781':
            fn = 'FlyingThings3D_subset_processed_35m/train/0016131'


        pc1 = os.path.join(fn,'pc1.npy')
        pc2 = os.path.join(fn,'pc2.npy')

        with open(pc1, 'rb') as fp:
            pos1 = np.load(fp)

        with open(pc2, 'rb') as fp2:
            pos2 = np.load(fp2)
        
        flow = pos2[:, :3] - pos1[:, :3]
        
        if self.train == 2:

            is_ground = np.logical_or(pos1[:,1] < -1.35, pos2[:,1] < -1.35)

            not_ground = np.logical_not(is_ground)

            near_mask = np.logical_and(pos1[:, 2] < 35, pos2[:, 2] < 35)
            near_mask = np.logical_and(not_ground, near_mask)

            indices = np.where(near_mask)[0]
        
        else:
            near_mask = np.logical_and(pos1[:, 2] < 35, pos2[:, 2] < 35)
            indices = np.where(near_mask)[0]## return the index of the truth condition

            
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

        if self.train == 2:
            return pos1, pos2, flow, fn
        else:
            return pos1, pos2, flow

    def __len__(self):
        return len(self.datapath)

    
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

if __name__ == '__main__':
    # import mayavi.mlab as mlab
    d = SceneflowDataset(npoints=2048)
    print(len(d))
    import time
    tic = time.time()
    for i in range(100):
        pc1, pc2, c1, c2, flow, m1, m2 = d[i]

        print(pc1.shape)
        print(pc2.shape)
        print(flow.shape)
        print(np.sum(m1))
        print(np.sum(m2))
        pc1_m1 = pc1[m1==1,:]
        pc1_m1_n = pc1[m1==0,:]
        print(pc1_m1.shape)
        print(pc1_m1_n.shape)
        mlab.points3d(pc1_m1[:,0], pc1_m1[:,1], pc1_m1[:,2], scale_factor=0.05, color=(1,0,0))
        mlab.points3d(pc1_m1_n[:,0], pc1_m1_n[:,1], pc1_m1_n[:,2], scale_factor=0.05, color=(0,1,0))
        raw_input()

        mlab.points3d(pc1[:,0], pc1[:,1], pc1[:,2], scale_factor=0.05, color=(1,0,0))
        mlab.points3d(pc2[:,0], pc2[:,1], pc2[:,2], scale_factor=0.05, color=(0,1,0))
        raw_input()
        mlab.quiver3d(pc1[:,0], pc1[:,1], pc1[:,2], flow[:,0], flow[:,1], flow[:,2], scale_factor=1)
        raw_input()

    print(time.time() - tic)
    print(pc1.shape, type(pc1))


