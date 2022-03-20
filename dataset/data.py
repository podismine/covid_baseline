#coding:utf8
import os
from torch.utils import data

import numpy as np
from sklearn.utils import shuffle
import nibabel as nib
import random
from random import gauss
from transformations import rotation_matrix
from scipy.ndimage.interpolation import map_coordinates
import glob

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def coordinateTransformWrapper(X_T1,maxDeg=0,maxShift=7.5,mirror_prob = 0.5):
    randomAngle = np.radians(maxDeg*2*(random.random()-0.5))
    unitVec = tuple(make_rand_vector(3))
    shiftVec = [maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5)]
    X_T1 = coordinateTransform(X_T1,randomAngle,unitVec,shiftVec)
    return X_T1

def coordinateTransform(vol,randomAngle,unitVec,shiftVec,order=1,mode='constant'):
    #from transformations import rotation_matrix
    ax = (list(vol.shape))
    ax = [ ax[i] for i in [1,0,2]]
    coords=np.meshgrid(np.arange(ax[0]),np.arange(ax[1]),np.arange(ax[2]))

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz=np.vstack([coords[0].reshape(-1)-float(ax[0])/2,     # x coordinate, centered
               coords[1].reshape(-1)-float(ax[1])/2,     # y coordinate, centered
               coords[2].reshape(-1)-float(ax[2])/2,     # z coordinate, centered
               np.ones((ax[0],ax[1],ax[2])).reshape(-1)])    # 1 for homogeneous coordinates
    
    # create transformation matrix
    mat=rotation_matrix(randomAngle,unitVec)

    # apply transformation
    transformed_xyz=np.dot(mat, xyz)

    # extract coordinates, don't use transformed_xyz[3,:] that's the homogeneous coordinate, always 1
    x=transformed_xyz[0,:]+float(ax[0])/2+shiftVec[0]
    y=transformed_xyz[1,:]+float(ax[1])/2+shiftVec[1]
    z=transformed_xyz[2,:]+float(ax[2])/2+shiftVec[2]
    x=x.reshape((ax[1],ax[0],ax[2]))
    y=y.reshape((ax[1],ax[0],ax[2]))
    z=z.reshape((ax[1],ax[0],ax[2]))
    new_xyz=[y,x,z]
    new_vol=map_coordinates(vol,new_xyz, order=order,mode=mode)
    return new_vol

from sklearn.model_selection import StratifiedKFold
def make_train_test(length, fold_idx, seed = 0, ns_splits = 5):
    assert 0 <= fold_idx and fold_idx < 5, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=ns_splits, shuffle=True, random_state=seed)
    labels = np.zeros((length))

    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    return train_idx, test_idx

class AllData(data.Dataset):

    def __init__(self, root, train=True):
        self.train = train
        self.root = root
        all_files = shuffle(sorted(glob.glob(os.path.join(self.root , "*"))), random_state = 1111) 
        all_files = [f for f in all_files if f.endswith(".nii.gz") or f.endswith(".npy")]

        train_idx, test_idx = make_train_test(len(all_files),0)

        if train:
            self.imgs = np.array(all_files)[train_idx]
            self.lbls1 = [float(f.split("/")[-1].split("_")[-2][0]) for f in self.imgs]
            self.lbls2 = [float(f.split("/")[-1].split("_")[-1][0]) for f in self.imgs]

            assert len(self.imgs) > 0, "No images found"
            print("Total files: ", len(self.imgs), sum(self.lbls1), sum(self.lbls2))
        else:
            self.imgs = np.array(all_files)[test_idx]
            self.lbls1 = [float(f.split("/")[-1].split("_")[-2][0]) for f in self.imgs]
            self.lbls2 = [float(f.split("/")[-1].split("_")[-1][0]) for f in self.imgs]

    def __getitem__(self,index):
        if self.imgs[index].endswith(".nii.gz"):
            img = nib.load(self.imgs[index]).get_fdata()
        elif self.imgs[index].endswith(".npy"):
            img = np.load(self.imgs[index])
        else:
            print("failed loading... Please check files.")
            exit()

        lbl1 = self.lbls1[index]
        lbl2 = self.lbls2[index]
        
        if self.train:
            #img = img #coordinateTransformWrapper(img,maxDeg=10,maxShift=5, mirror_prob = 0)
            img = coordinateTransformWrapper(img,maxDeg=10,maxShift=5, mirror_prob = 0)
        else:
            img = img

        img = img[np.newaxis,...]

        return img, lbl1, lbl2
    
    def __len__(self):
        return len(self.imgs)

import torch
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target1, self.next_target2 = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target1 = None
            self.next_target2 = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target1 = self.next_target1.cuda(non_blocking=True).float()
            self.next_target2 = self.next_target2.cuda(non_blocking=True).float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target1 = self.next_target1
        target2 = self.next_target2

        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target1 is not None:
            target1.record_stream(torch.cuda.current_stream())
        if target2 is not None:
            target2.record_stream(torch.cuda.current_stream())
        self.preload()

        return input, target1, target2
