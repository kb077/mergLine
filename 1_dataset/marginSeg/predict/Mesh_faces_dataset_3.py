from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix



import torch.utils.data as data
from PIL import Image
import h5py
import numpy as np
import torch

def is_hdf5_file(filename):
    return filename.lower().endswith('.h5')


def get_keys(hdf5_path):
    with h5py.File(hdf5_path, 'r') as file:
        return list(file.keys())

class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, num_classes=15, patch_size=7000):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hdf5_path = data_list_path
        self.keys = get_keys(self.hdf5_path)

        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hdf5_file = h5py.File(self.hdf5_path, "r")
        slide_data = hdf5_file[self.keys[idx]]
        X = slide_data['cells'][()]
        barycenters = slide_data['center'][()]
        normals = slide_data['normals'][()]
        face_normals = slide_data['face_normals'][()]
        
        #normalized data
        maxs = X.max(axis=0)
        mins = X.min(axis=0)
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)
        nmaxs = normals.max(axis=0)
        nmins = normals.min(axis=0)
        
        nfmeans = face_normals.mean(axis=0)
        nfstds = face_normals.std(axis=0)
        nfmaxs = face_normals.max(axis=0)
        nfmins = face_normals.min(axis=0)
        
        X = (X-mins)/(maxs-mins)
        normals = (normals-nmins)/(nmaxs-nmins)
        face_normals = (face_normals-nfmins)/(nfmaxs-nfmins)
        X = np.column_stack((X, barycenters, normals,face_normals))
        Y = slide_data['face_label'][()]
        Y = Y.reshape(len(Y),1)
        labels = Y

        # initialize batch of input and label
        X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
        Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')
        S1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        S2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')

        # calculate number of valid cells (tooth instead of gingiva)
        positive_idx = np.argwhere(labels>=0)[:, 0] #tooth idx
        negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx

        num_positive = len(positive_idx) # number of selected tooth cells

        if num_positive > self.patch_size: # all positive_idx in this patch
            positive_selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
            selected_idx = positive_selected_idx
        else:   # patch contains all positive_idx and some negative_idx
            num_negative = self.patch_size - num_positive # number of selected gingiva cells
            positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
            negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
            selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

        selected_idx = np.sort(selected_idx, axis=None)

        X_train[:] = X[selected_idx, :]
        Y_train[:] = Y[selected_idx, :]


        X_train = X_train.transpose(1, 0)
        Y_train = Y_train.transpose(1, 0)


        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train)}
        return sample
