import numpy as np
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl 
import torchvision.models as models
from torch.nn.functional import softmax
import torchmetrics
import utils


# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)

from pytorch3d.structures import Meshes

from sklearn.metrics import classification_report, confusion_matrix
from pytorch3d.vis.plotly_vis import plot_scene

def GetView(meshes,phong_renderer,R,T):

    R = R.to(torch.float32)
    T = T.to(torch.float32)
    
    fragments = phong_renderer.rasterizer(meshes.clone(),R=R,T=T)
    pix_to_face = fragments.pix_to_face   
    pix_to_face = pix_to_face.permute(0,3,1,2)
    return pix_to_face

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)

        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output



class SelfAttention(nn.Module):
    def __init__(self,in_units,out_units):
        super().__init__()


        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
    
    def forward(self, query, values):        
        score = self.Sigmoid(self.V(self.Tanh(self.W1(query))))
        
        attention_weights = score/torch.sum(score, dim=1,keepdim=True)

        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

class Fly_by_CNN(pl.LightningModule):
    def __init__(self, radius, ico_lvl, dropout_lvl, batch_size, num_classes, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=58)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=58)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=58)
        self.radius = radius
        self.ico_lvl = ico_lvl
        self.dropout_lvl = dropout_lvl
        self.image_size = 224
        self.batch_size = batch_size
        self.num_classes = num_classes
        #ico_sphere, _a, _v = utils.RandomRotation(utils.CreateIcosahedron(self.radius, ico_lvl))
        ico_sphere = utils.CreateIcosahedron(self.radius, ico_lvl)
        ico_sphere_verts, ico_sphere_faces, self.ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_faces = ico_sphere_faces
        self.ico_sphere_edges = np.array(self.ico_sphere_edges)
        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_pos = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_pos)
            T_current = -torch.bmm(R_current.transpose(1,2), camera_pos[:,:,None])[:,:,0]
            R.append(R_current)
            T.append(T_current)

        self.R = torch.cat(R)
        self.T = torch.cat(T)
        self.R = self.R.to(torch.float32)
        self.T = self.T.to(torch.float32)

        efficient_net = models.efficientnet_b0(pretrained=True)
        efficient_net.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        efficient_net.classifier = Identity()

        self.drop = nn.Dropout(p=dropout_lvl)
        self.TimeDistributed = TimeDistributed(efficient_net)

        self.WV = nn.Linear(1280, 512)

        #conv2dForQuery = nn.Conv2d(1280, 1280, kernel_size=(3,3),stride=2,padding=0) #1280,512
        #conv2dForValues = nn.Conv2d(512, 512, kernel_size=(3,3),stride=2,padding=0)  #512,512

        #self.IcosahedronConv2dForQuery = IcosahedronConv2d(conv2dForQuery,self.ico_sphere_verts,self.ico_sphere_edges)
        #self.IcosahedronConv2dForValues = IcosahedronConv2d(conv2dForValues,self.ico_sphere_verts,self.ico_sphere_edges)

        self.Attention = SelfAttention(1280, 128)
        self.Classification = nn.Linear(512, num_classes)


        self.Sigmoid = nn.Sigmoid()
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=58)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=58)


        self.cameras = FoVPerspectiveCameras()

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0, 
            faces_per_pixel=1, 
            max_faces_per_bin=100000
        )
        # We can add a point light in front of the object.

        lights = AmbientLights()
        rasterizer = MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            )
        self.phong_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=self.cameras, lights=lights)
        )

    def forward(self, x):
        V, F, VF, FF = x
        V = V.to(self.device)
        F = F.to(self.device)
        VF = VF.to(self.device)
        FF = FF.to(self.device)
        x, PF = self.render(V,F,VF,FF)
        
        query = self.TimeDistributed(x)
        values = self.WV(query)
        x_a, w_a = self.Attention(query, values)
        x_a = self.drop(x_a)
        x = self.Classification(x_a)
        
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def render(self, V, F, VF, FF):
        textures = TexturesVertex(verts_features=VF)
        V = V.to(torch.float32)
        F = F.to(torch.float32)
        meshes = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )
        fig2 = plot_scene({
                    "display of all fibers": {
                        "mesh": meshes,
                    }
                })
        #fig2.show()

        phong_renderer = self.phong_renderer.to(self.device)
        PF = []
        for i in range(len(self.R)):
            R = self.R[i][None].to(self.device)
            T = self.T[i][None].to(self.device)
            pixel_to_face = GetView(meshes,phong_renderer,R,T)
            PF.append(pixel_to_face.unsqueeze(dim=1))

        PF = torch.cat(PF,dim=1)

        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,:,index],PF)*(PF >= 0)) # take each feature
        x = torch.cat(l_features,dim=2)

        return x, PF

    def training_step(self, train_batch, train_batch_idx):

        V, F, VF, FF, labels = train_batch
        labels = labels.squeeze(dim=1)
        x = self((V, F, VF, FF))

        x = self.Sigmoid(x).squeeze(dim=1)
        loss = self.loss(x, labels)

        self.log('train_loss', loss, batch_size=self.batch_size)
        self.log('train_accuracy', self.train_accuracy, batch_size=self.batch_size)

        return loss

        
    def validation_step(self, val_batch, val_batch_idx):

        V, F, VF, FF, labels = val_batch
        labels = labels.squeeze(dim=1)
        x = self((V, F, VF, FF))

        x = self.Sigmoid(x).squeeze(dim=1)
        loss = self.loss(x, labels)

        self.log('val_loss', loss, batch_size=self.batch_size)
        predictions = torch.argmax(x, dim=1)
        self.val_accuracy(predictions.reshape(-1,1), labels.reshape(-1,1))
        self.log('val_accuracy', self.val_accuracy, batch_size=self.batch_size)

        return loss

    def test_step(self, test_batch, test_batch_idx):
        V, F, VF, FF, labels = test_batch
        labels = labels.squeeze(dim=1)
        x = self((V, F, VF, FF))
        x = self.Sigmoid(x).squeeze(dim=1)

        loss = self.loss(x, labels)
        self.log('test_loss', loss, batch_size=self.batch_size)
        self.log('test_accuracy', self.val_accuracy, batch_size=self.batch_size)
        
        predictions = torch.argmax(x, dim=1)

        output = [predictions, labels]
        return output

    def test_epoch_end(self, outputs):
        y_pred = []
        y_true = []
        
        for output in outputs:
            y_pred.append(output[0].tolist())
            y_true.append(output[1].tolist())

        y_pred = [ele for sousliste in y_pred for ele in sousliste]
        y_true = [ele for sousliste in y_true for ele in sousliste]
        
        
        y_pred = [[int(ele)] for ele in y_pred]
        
        print("y_pred", y_pred)
        print("y_true", y_true)
        
        print(classification_report(y_true, y_pred))