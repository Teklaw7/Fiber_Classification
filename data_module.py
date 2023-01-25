from torch.utils.data import Dataset, DataLoader
import torch
#import split_train_eval
import utils
import pytorch_lightning as pl 
import vtk
from pytorch3d.structures import Meshes
from random import *
from pytorch3d.vis.plotly_vis import plot_scene
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from pytorch3d.renderer import TexturesVertex
from torch.utils.data._utils.collate import default_collate
import pandas as pd

class Bundles_Dataset(Dataset):
    def __init__(self, data, path_data, path_ico, transform=True, column_class='class',column_id='id', column_label='label'):
        self.data = data
        self.transform = transform
        self.path_data = path_data
        self.path_ico = path_ico
        self.column_class = column_class
        self.column_id = column_id
        self.column_label = column_label

     
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_row = self.data.loc[idx]
        sample_id = sample_row[self.column_id]
        sample_class = sample_row[self.column_class]
        sample_label = sample_row[self.column_label]
        
        path_cc1 = f"/CMF/data/timtey/tracts/archives/{sample_id}_tracts/{sample_class}.vtp"
        print("path_cc1",path_cc1)
        
        x_min, x_max, y_min, y_max, z_min, z_max = bounding_box(sample_id)
        cc1 = utils.ReadSurf(path_cc1)
        n = randint(0,cc1.GetNumberOfCells()-1)
        cc1_extract = utils.ExtractFiber(cc1,n)
        cc1_tf=vtk.vtkTriangleFilter()
        cc1_tf.SetInputData(cc1_extract)
        cc1_tf.Update()
        cc1_extract_tf = cc1_tf.GetOutput()
        verts, faces, edges = utils.PolyDataToNumpy(cc1_extract_tf)
        faces = torch.tensor(faces)
        verts = torch.tensor(verts)
        
        #Transformations 
        if self.transform:        
            for i in range(len(verts)):
                verts[i][0] = (2*(verts[i][0] - x_min)/(x_max - x_min)) - 1
                verts[i][1] = (2*(verts[i][1] - y_min)/(y_max - y_min)) - 1
                verts[i][2] = (2*(verts[i][2] - z_min)/(z_max - z_min)) - 1
        
        
        EstimatedUncertainty = cc1_extract_tf.GetPointData().GetScalars("EstimatedUncertainty")
        FA1 = cc1_extract_tf.GetPointData().GetScalars("FA1")
        FA2 = cc1_extract_tf.GetPointData().GetScalars("FA2")
        HemisphereLocataion = cc1_extract_tf.GetPointData().GetScalars("HemisphereLocataion")
        cluster_idx = cc1_extract_tf.GetPointData().GetScalars("cluster_idx")
        trace1 = cc1_extract_tf.GetPointData().GetScalars("trace1")
        trace2 = cc1_extract_tf.GetPointData().GetScalars("trace2")
        vtkOriginalPointIds = cc1_extract_tf.GetPointData().GetScalars("vtkOriginalPointIds")
        TubeNormals = cc1_extract_tf.GetPointData().GetScalars("TubeNormals")

        EstimatedUncertainty = vtk_to_numpy(EstimatedUncertainty)
        FA1 = vtk_to_numpy(FA1)
        FA2 = vtk_to_numpy(FA2)
        HemisphereLocataion = vtk_to_numpy(HemisphereLocataion)
        cluster_idx = vtk_to_numpy(cluster_idx)
        trace1 = vtk_to_numpy(trace1)
        trace2 = vtk_to_numpy(trace2)
        vtkOriginalPointIds = vtk_to_numpy(vtkOriginalPointIds)
        TubeNormals = vtk_to_numpy(TubeNormals)

        TubeNormals_idx = []
        TubeNormals_idy = []
        TubeNormals_idz = []
        for i in range(len(TubeNormals)):
            TubeNormals_idx.append(TubeNormals[i][0])
            TubeNormals_idy.append(TubeNormals[i][1])
            TubeNormals_idz.append(TubeNormals[i][2])

        EstimatedUncertainty = torch.tensor([float(element) for element in EstimatedUncertainty])
        FA1 = torch.tensor([float(element) for element in FA1])
        FA2 = torch.tensor([float(element) for element in FA2])
        HemisphereLocataion = torch.tensor([float(element) for element in HemisphereLocataion])
        cluster_idx = torch.tensor([float(element) for element in cluster_idx])
        trace1 = torch.tensor([float(element) for element in trace1])
        trace2 = torch.tensor([float(element) for element in trace2])
        vtkOriginalPointIds = torch.tensor([float(element) for element in vtkOriginalPointIds])
        TubeNormals_idx = torch.tensor([float(element) for element in TubeNormals_idx])
        TubeNormals_idy = torch.tensor([float(element) for element in TubeNormals_idy])
        TubeNormals_idz = torch.tensor([float(element) for element in TubeNormals_idz])

        l_features = [TubeNormals_idx.unsqueeze(dim=1), TubeNormals_idy.unsqueeze(dim=1), TubeNormals_idz.unsqueeze(dim=1)]
        #EstimatedUncertainty.unsqueeze(dim=1), FA1.unsqueeze(dim=1), FA2.unsqueeze(dim=1), HemisphereLocataion.unsqueeze(dim=1), cluster_idx.unsqueeze(dim=1), trace1.unsqueeze(dim=1), trace2.unsqueeze(dim=1), vtkOriginalPointIds.unsqueeze(dim=1), 
        
        vertex_features = torch.cat(l_features, dim=1)

        faces_pid0 = faces[:,0:1]
        nb_faces = len(faces)
        offset = torch.zeros((nb_faces,vertex_features.shape[1]), dtype=int) + torch.Tensor([i for i in range(vertex_features.shape[1])]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])      
        
        face_features = torch.take(vertex_features,faces_pid0_offset)
        
        
        vertex_features_mesh=vertex_features.unsqueeze(dim=0)
        
        texture =TexturesVertex(verts_features=vertex_features_mesh)
        mesh = Meshes(verts=[verts], faces=[faces], textures=texture) 
        
        

        ### labels ###
        labels = torch.tensor([sample_label])
        

        #Load  Icosahedron
        reader = utils.ReadSurf(self.path_ico)
        verts_ico, faces_ico, edges_ico = utils.PolyDataToTensors(reader)
        nb_faces = len(faces_ico)
        
        
        print("labels are back", labels)
        return verts, faces,vertex_features,face_features, labels

def bounding_box(sample_id):

    atlases_path="/CMF/data/timtey/UKF"
    path_atlas = f"{atlases_path}/{sample_id}_ukf.vtk"
    atlas = utils.ReadSurf(path_atlas)

    min_max=atlas.GetBounds()
    xmin=min_max[0]
    xmax=min_max[1]
    ymin=min_max[2]
    ymax=min_max[3]
    zmin=min_max[4]
    zmax=min_max[5]
    
    return xmin, xmax, ymin, ymax, zmin, zmax




class Bundles_DataModule(pl.LightningDataModule):
    def __init__(self, path_data, path_ico, batch_size, train_path, val_path, test_path, transform=True):
        super().__init__()
        self.path_data = path_data
        self.path_ico = path_ico
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.transform = transform

    def setup(self, stage=None):

        #list_train_data = open(self.train_path, "r").read().splitlines()
        #list_val_data = open(self.val_path, "r").read().splitlines()
        #list_test_data = open(self.test_path, "r").read().splitlines()
        list_train_data = pd.read_csv(self.train_path)
        list_val_data = pd.read_csv(self.val_path)
        list_test_data = pd.read_csv(self.test_path)

        self.train_dataset = Bundles_Dataset(list_train_data, self.path_data, self.path_ico, self.transform)
        self.val_dataset = Bundles_Dataset(list_val_data, self.path_data, self.path_ico, self.transform)
        self.test_dataset = Bundles_Dataset(list_test_data, self.path_data, self.path_ico, self.transform)
        

    
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=custom_collate, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=custom_collate, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=custom_collate, shuffle=True)
    



def custom_collate(batch):

    liste_verts = []
    liste_faces = []
    for idx in range(len(batch)):
        liste_verts.append(batch[idx][2].shape[0])    
        liste_faces.append(batch[idx][3].shape[0])

    max_verts_features_verts = max(liste_verts)
    max_verts_features_faces = max(liste_faces)
    
    for idx in range(len(batch)):
        v= batch[idx][0]
        f= batch[idx][1]
        vf= batch[idx][2]
        ff= batch[idx][3]
        labels = batch[idx][4]

        while len(v) < max_verts_features_verts :
            v = torch.cat((v, torch.zeros(1,3)), dim=0)

        while len(f) < max_verts_features_faces :
            f = torch.cat((f, torch.zeros((1,3))), dim=0)

        while len(vf) < max_verts_features_verts :
            vf = torch.cat((vf, torch.zeros((1,3))), dim=0)

        while len(ff) < max_verts_features_faces :
            ff = torch.cat((ff, torch.zeros((1,3))), dim=0)

        batch[idx]=(v,f,vf,ff,labels)

    return default_collate(batch)