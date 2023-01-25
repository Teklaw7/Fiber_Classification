from __future__ import print_function
import argparse
import configparser
import glob
import json
import os
from os import path as osp
from os.path import basename as osbn
from time import time

import ants
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#from torch_geometric.data import Batch as gBatch
#from torch_geometric.data import DataListLoader as gDataLoader

#from sDEC import DECSeq
#import datasets as ds
#from utils import ReadSurf , PolyDataToNumpy
import utils
import pytorch3d
import pytorch3d.renderer as pyr
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.vis.plotly_vis import plot_scene
import vtk
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

from data_module import Bundles_Dataset, Bundles_DataModule

from brain_module_cnn import Fly_by_CNN

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import pandas as pd
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def csv_changes(csv_path_train, csv_path_valid, csv_path_test):
    df_train = pd.read_csv(csv_path_train)
    df_valid = pd.read_csv(csv_path_valid)
    df_test = pd.read_csv(csv_path_test)
    
    sample_class_csv = df_test['class'].unique()
    df_train['label'] = df_train['class']
    df_valid['label'] = df_valid['class']
    df_test['label'] = df_test['class']


    for i in range(len(sample_class_csv)):
        df_train['label'] = df_train['label'].replace(sample_class_csv[i], i+1)
        df_valid['label'] = df_valid['label'].replace(sample_class_csv[i], i+1)
        df_test['label'] = df_test['label'].replace(sample_class_csv[i], i+1)

    path_train_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_train_label_to_number.csv"
    path_valid_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_valid_label_to_number.csv"
    path_test_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_test_label_to_number.csv"

    df_train.to_csv(path_train_final)
    df_valid.to_csv(path_valid_final)
    df_test.to_csv(path_test_final)
    
    return path_train_final, path_valid_final, path_test_final


### Part Training ###

### Training call ###

num_classes = 58
nb_epochs = 25
nb_iteration=20
sample_size=1000
batch_size=20
cfg={'fixed_size': sample_size, 'same_size': True, 'save_path': 'model.pth'}
dropout_lvl=0.1
radius=1
ico_lvl=1
min_delta_early_stopping = 0.00
patience_early_stopping=10

path_data="/CMF/data/timtey/tracts/archives"
path_ico = "/NIRAL/tools/atlas/Surface/Sphere_Template/sphere_f327680_v163842.vtk"
train_path="/CMF/data/timtey/tracts/tracts_filtered_train_train.csv"
val_path="/CMF/data/timtey/tracts/tracts_filtered_train_valid.csv"
test_path="/CMF/data/timtey/tracts/tracts_filtered_test.csv"
path_train_final, path_valid_final, path_test_final = csv_changes(train_path, val_path, test_path)
#path_test_final = "/home/timtey/Documents/Projet/dataset/tracts_filtered_train_test_label_to_number_copy.csv"
brain_data=Bundles_DataModule(path_data, path_ico, batch_size, path_train_final, path_valid_final, path_test_final)

model= Fly_by_CNN(radius, ico_lvl, dropout_lvl, batch_size, num_classes, learning_rate=0.001)

early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=min_delta_early_stopping, patience=patience_early_stopping, verbose=True, mode='min')
trainer=Trainer(max_epochs=nb_epochs, accelerator="gpu")

trainer.fit(model, brain_data)

trainer.test(model, brain_data)


