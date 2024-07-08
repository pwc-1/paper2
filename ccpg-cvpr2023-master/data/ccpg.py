import os
import cv2
import numpy as np
import mindspore
import pickle
from math import ceil

class CCPG_DataSet():
    def __init__(self, dataset_path, mode="Train"):
        self.img_path = dataset_path
        self.mode = mode
        pkls, ids, cloths, views = self.get_img_paths_labels(self.img_path)
        self.data = pkls
        self.ids = ids
        self.cloths = cloths
        self.views = views
        self.num = len(pkls)

    def __getitem__(self, index):
        pkl_path = self.data[index]
        pkl_path = open(pkl_path,'rb')
        vid = pickle.load(pkl_path)
        vid = vid[:, :, 20:-20]
        time_length = vid.shape[0]
        _idx = np.array(list(range(time_length)))
        if self.mode == "Train":   
            if time_length < 30:
                _idx = _idx.repeat(ceil(30/time_length))
            _idx = np.random.choice(_idx, 30, replace=False)
        return vid[_idx, :, :], self.ids[index], self.cloths[index], self.views[index]
    
    def __len__(self):
        return self.num 

    def get_img_paths_labels(self, dataset_path):
        vid_path = []
        id_labels = []
        cloth_labels = []
        views_labels = []
        for ids in list(range(100)):
            ids += 100
            id_path = os.path.join(dataset_path, str(ids))
            for cloth in os.listdir(id_path):
                cloth_path = os.path.join(id_path, cloth)
                for view in os.listdir(cloth_path):
                    view_path = os.path.join(cloth_path, view)
                    for seq in os.listdir(view_path):
                        seq_path = os.path.join(view_path, seq)
                        for pkl in os.listdir(seq_path):
                            pkl_path = os.path.join(seq_path, pkl)
                            vid_path.append(pkl_path)
                            id_labels.append(ids-100)
                            cloth_labels.append(cloth)
                            views_labels.append(view)
        return vid_path, id_labels, cloth_labels, views_labels
