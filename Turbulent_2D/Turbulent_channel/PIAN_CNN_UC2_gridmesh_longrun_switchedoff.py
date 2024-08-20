 
import torch
import torchphysics as tp
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from pytorch_lightning import loggers as pl_loggers

#Input
X = tp.spaces.R1('x')
Y = tp.spaces.R1('y')
C = tp.spaces.R1("c") # number of case
#output
U = tp.spaces.R1('u')
V = tp.spaces.R1('v')
URMS = tp.spaces.R1('urms')
VRMS = tp.spaces.R1('vrms')
UV=tp.spaces.R1('uv')
P=tp.spaces.R1('p')


# GAN macros
GPU="cuda:0"
GAN_weight=1 # Generator weight
L_x=3.6 # length of domain
N_x=240 # grid in x
N_x_sub=100
N_y=76 # grid in y
N_dists=1 # N 1d roughness for fake images during training
GP_weight=10
N_epochs=1000
Grid_data=False



X_interval = tp.domains.Interval(X, 0, L_x) # <-add the bounds of the Interval (0, 2)
Y_interval = tp.domains.Interval(Y, 0, 1.0)
C_interval = tp.domains.Interval(C,0,288) # number of available roughness profiles

Sim_domain = X_interval*Y_interval*C_interval


Y_interval_sub_low=tp.domains.Interval(Y, 0, 0.15)
Sim_domain_sub_low=X_interval*Y_interval_sub_low*C_interval
#Y_interval_sub_up=tp.domains.Interval(Y, 2-0.15, 2)
#Sim_domain_sub_up=X_interval*Y_interval_sub_up*C_interval



DF_Data=pd.read_csv("Data/Flow_3d_2289_6_Cx3_Cy3X240Y76_half.csv")
DF=DF_Data[["U","V","urms","vrms","uv",]]

N_c=len(DF_Data["c"].unique()) # number of available training data

#Data_Pinn=torch.tensor(DF_Data[["U","V","urms","vrms","uv"]].to_numpy().transpose((1,0)).reshape((5,N_c,N_x,N_y)),dtype=torch.float32,device=GPU)
Data_Pinn=torch.permute(torch.tensor(DF_Data[["U","V","urms","vrms","uv"]].to_numpy(),dtype=torch.float32,device=GPU).reshape((N_c,N_x,N_y,5)),(0,3,1,2))


#eta=torch.cos(torch.pi*(torch.tensor([i for i in range(N_y*2)],requires_grad=False,device=GPU))/(N_y*2-1)) 

if Grid_data:
    ys=torch.linspace(0,1,N_y,dtype=torch.float32,device=GPU)
else:
    eta=torch.cos(torch.pi*(torch.tensor([i for i in range(N_y*2)],device=GPU))/(N_y*2-1)) 
    ys=(1-eta.detach())[:N_y]

DF_DIST=pd.read_csv("Data/dist_2289_1Ds.csv",header=None)
x_corr=DF_DIST.iloc[0].to_numpy()
dist=DF_DIST.iloc[1:].to_numpy()
x_corr_new=np.linspace(min(x_corr),max(x_corr),N_x)
dist_new=np.zeros((len(dist),N_x))
for i in range(len(dist)):
    dist_new[i,:]=np.interp(x_corr_new,x_corr,dist[i])
x_corr_new=torch.tensor(x_corr_new,requires_grad=False, dtype=torch.float32,device=GPU)
x_corr=torch.tensor(x_corr,requires_grad=False, dtype=torch.float32,device=GPU)
dist_new=torch.tensor(dist_new,requires_grad=False, dtype=torch.float32,device=GPU)
dist=torch.tensor(dist, requires_grad=False,dtype=torch.float32,device=GPU)


ymesh=ys.reshape((1,1,1,-1)).expand((N_c,1,N_x,-1))

matrix_mask_Data=(ymesh<dist_new.reshape((N_c,1,N_x,-1)))|(ymesh>=(2-dist_new.reshape((N_c,1,N_x,-1))))
matrix_mask_Data=matrix_mask_Data.long()







repo_dist=dist
repo_dist_new=dist_new
#np.zeros((len(repo_dist),N_x))
#for i in range(len(repo_dist)):
#    repo_dist_new[i,:]=np.interp(x_corr_new,x_corr,repo_dist[i])


def IBM_irr_filter_low(x,y,c):
    position_r=torch.tensor([sum(x_corr<x_i)-1 for x_i in x],device=GPU)
    #position_r=sum(x_corr<x)
    position_l=position_r-1
    x_new=torch.transpose(x,1,0)[0]
    c_new=torch.transpose(c,1,0)[0]
    c_new = c_new.to(torch.long)
    height=dist[c_new,position_l]+(dist[c_new,position_r]-dist[c_new,position_l])*(x_new-x_corr[position_l])/(x_corr[position_r]-x_corr[position_l])
    return (y[...,0]<=height)
IBM_sampler_irr_low = tp.samplers.RandomUniformSampler(Sim_domain_sub_low,n_points=5000,filter_fn=IBM_irr_filter_low).make_static(resample_interval=2000)




def Inner_filter(x,y,c):
    position_r=torch.tensor([sum(x_corr<x_i)-1 for x_i in x],device=GPU)
    #position_r=sum(x_corr<x)
    position_l=position_r-1
    x_new=torch.transpose(x,1,0)[0]
    c_new=torch.transpose(c,1,0)[0]
    c_new = c_new.to(torch.long)
    height=dist[c_new,position_l]+(dist[c_new,position_r]-dist[c_new,position_l])*(x_new-x_corr[position_l])/(x_corr[position_r]-x_corr[position_l])
    return (y[...,0]>height)
inner_sampler = tp.samplers.RandomUniformSampler(Sim_domain, n_points=5000,filter_fn=Inner_filter).make_static(resample_interval=2000)#,filter_fn=Inner_filter)






bound_sampler_low = tp.samplers.RandomUniformSampler(X_interval*Y_interval.boundary_left*C_interval, n_points=250)

bound_sampler_up = tp.samplers.RandomUniformSampler(X_interval*Y_interval.boundary_right*C_interval, n_points=500)





def self_sin(input):
    return input.sin()
def self_cos(input):
    return input.cos()


import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(seed=42)
class ResidualBlock_1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock_1d, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    

class ResNet1d(nn.Module):
    def __init__(self, block, layers,input_space,output_space,N_features,sigma_1=1,sigma_2=15):
        super(ResNet1d, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv1d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm1d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 64, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 128, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 128, layers[3], stride = 2)
        self.avgpool = nn.AvgPool1d(2, stride=0)
        #self.act_binary=nn.Sigmoid()
        #self.fc_Res = nn.Linear(2048, 2)

        self.W_1 = torch.tensor(torch.randn(input_space.dim , N_features //2, dtype=torch.float32,device=GPU) * sigma_1, dtype=torch.float32, requires_grad=False)
        self.W_2 = torch.tensor(torch.randn(input_space.dim , N_features //2, dtype=torch.float32,device=GPU) * sigma_2, dtype=torch.float32, requires_grad=False)
        self.register_buffer("selfW1", self.W_1, persistent=False)
        self.register_buffer("selfW2", self.W_2, persistent=False)
        self.output_space=output_space
        self.fc1_l=nn.Linear(in_features=N_features,out_features=150)
        self.fc2_l=nn.Linear(in_features=150,out_features=150)
        self.fc3_l=nn.Linear(in_features=150,out_features=150)
        ###
        self.fc1_r=nn.Linear(in_features=N_features,out_features=150)
        self.fc2_r=nn.Linear(in_features=150,out_features=150)
        self.fc3_r=nn.Linear(in_features=150,out_features=150)


        self.fc_combo1=nn.Linear(in_features=2944+300,out_features=2048)
        self.fc_combo2=nn.Linear(in_features=2048,out_features=2048)
        self.fc_combo3=nn.Linear(in_features=2048,out_features=2048)

        self.out=nn.Linear(in_features=2048,out_features=output_space.dim)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self,x,t):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

    #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = nn.ReLU(self.fc_Res(x))
        t=t.as_tensor[:,0:2]
        t_1=torch.concat((self_sin(torch.matmul(t,self.W_1)),self_cos(torch.matmul(t,self.W_1))),1)
        t_1=self_sin(self.fc1_l(t_1))
        t_1=self_sin(self.fc2_l(t_1))
        t_1=self_sin(self.fc3_l(t_1))
        t_2=torch.concat((self_sin(torch.matmul(t,self.W_2)),self_cos(torch.matmul(t,self.W_2))),1)
        t_2=self_sin(self.fc1_r(t_2))
        t_2=self_sin(self.fc2_r(t_2))
        t_2=self_sin(self.fc3_r(t_2))

        t=torch.concat((t_1,t_2,x),1)
        t=self_sin(self.fc_combo1(t))
        t=self_sin(self.fc_combo2(t))
        t=self_sin(self.fc_combo3(t))
        t=self.out(t)

        return tp.problem.spaces.Points(t, self.output_space)
        
model = ResNet1d(ResidualBlock_1d,[2,2,2,2],input_space=X*Y,output_space=U*V*URMS*VRMS*UV*P,N_features=300).to(GPU)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(6, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        #self.act_binary=nn.Sigmoid()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 1)
        self.relu=nn.ReLU()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

    #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x
disc=ResNet(ResidualBlock, [2, 2, 2, 2]).to(GPU)



def pde_IBM(u,v):
    return torch.sqrt(torch.square(u)+torch.square(v))+torch.abs(urms)+torch.abs(vrms)+torch.abs(uv)
pde_cond_IBM_low = tp.conditions.PINNCondition_CNN(model, IBM_sampler_irr_low, pde_IBM, dist_matrix=dist,weight=50,name='IBM_low')


#def pde_IBM_uu(urms,vrms,uv,p):
#    return torch.abs(urms)+torch.abs(vrms)+torch.abs(uv)
#pde_cond_IBM_low_uu = tp.conditions.PINNCondition_CNN(model, IBM_sampler_irr_low, pde_IBM_uu, dist_matrix=dist,weight=50,name='IBM_low_uu')








def pde_mass(u,v,x,y):
    return tp.utils.grad(u,x)+tp.utils.grad(v,y)-0.0
pde_cond_mass = tp.conditions.PINNCondition_CNN(model, inner_sampler,pde_mass, dist_matrix=dist, weight=50,name='Conti')



def pde_residual_x(u,v, x, y,urms,vrms,uv,p,c):
    c_new=torch.transpose(c,1,0)[0]
    c_new = c_new.to(torch.long)
    residual_momentum= u*tp.utils.grad(u,x)+ v*tp.utils.grad(u,y)+(-tp.utils.laplacian(u, x)-tp.utils.laplacian(u,y))/6000 - ( (2/15)**2/((1-torch.mean(dist[c_new,:]))**3) ) +tp.utils.grad(torch.square(urms/6),x)+tp.utils.grad(uv/60,y)+tp.utils.grad(p,x)
    return residual_momentum

pde_cond_x = tp.conditions.PINNCondition_CNN(model, inner_sampler, pde_residual_x, dist_matrix=dist,weight=51,name='Momentum_x')



def pde_residual_y(u,v, x, y,urms,vrms,uv,p):
    residual_momentum= u*tp.utils.grad(v,x)+v*tp.utils.grad(v,y)+(-tp.utils.laplacian(v, x)-tp.utils.laplacian(v,y))/6000+tp.utils.grad(uv/60,x)+tp.utils.grad(torch.square(vrms/6),y)+tp.utils.grad(p,y)
    return residual_momentum

pde_cond_y = tp.conditions.PINNCondition_CNN(model, inner_sampler, pde_residual_y, dist_matrix=dist,weight=50,name='Momentum_y')

#def boundary_residual_x(u, x,y):
#    return torch.square(u) - 0.0
#boundary_cond_x = tp.conditions.PINNCondition_CNN(model, bound_sampler_low, boundary_residual_x, dist_matrix=dist, weight=1,name='noslip_x')

#def boundary_residual_y(v, x,y):
#    return torch.square(v) - 0.0

#boundary_cond_y = tp.conditions.PINNCondition_CNN(model, bound_sampler_low, boundary_residual_y, dist_matrix=dist, weight=1,name='noslip_y')

#def boundary_residual_uu(urms, x,y):
#    return torch.square(urms) - 0.0

#boundary_cond_uu = tp.conditions.PINNCondition_CNN(model, bound_sampler_low, boundary_residual_uu, dist_matrix=dist, weight=1,name='noslip_x_uu')

#def boundary_residual_vv(vrms, x,y):
#    return torch.square(vrms) - 0.0

#boundary_cond_vv = tp.conditions.PINNCondition_CNN(model, bound_sampler_low, boundary_residual_vv, dist_matrix=dist, weight=1,name='noslip_x_vv')




#def boundary_residual_uv(uv, x,y):
#    return uv - 0.0

#boundary_cond_uv = tp.conditions.PINNCondition_CNN(model, bound_sampler_low, boundary_residual_uv, dist_matrix=dist, weight=1,name='noslip_x_uv')

def boundary_residual(u,v,urms,vrms,uv,p, x,y,c):
    un=torch.abs(tp.utils.grad(u,y))+torch.abs(v)+torch.abs(tp.utils.grad(urms,y))+torch.abs(tp.utils.grad(vrms,y))+torch.abs(uv)+torch.abs(p)
    return un
boundary_cond_up = tp.conditions.PINNCondition_CNN(model, bound_sampler_up, boundary_residual, dist_matrix=dist, weight=1,name='bound_up')

#def boundary_residual_y_grad(v):
#    return torch.abs(v)
#boundary_cond_y_up = tp.conditions.PINNCondition_CNN(model, bound_sampler_up, boundary_residual_y_grad, dist_matrix=dist, #weight=1,name='bound_y_up')


#def boundary_residual_Re_grad(urms,vrms, x,y,c):
#    uun=tp.utils.grad(urms,y)
#    vvn=tp.utils.grad(vrms,y)
#    return torch.abs(uun)+torch.abs(vvn)
#boundary_cond_Re_up = tp.conditions.PINNCondition_CNN(model, bound_sampler_up, boundary_residual_Re_grad, dist_matrix=dist, weight=1,name='bound_Re_up')



#def boundary_residual_uv_grad(uv, x,y,c):
#    return torch.abs(uv)
#boundary_cond_uv_up = tp.conditions.PINNCondition_CNN(model, bound_sampler_up, boundary_residual_uv_grad, dist_matrix=dist, weight=1,name='bound_uv_up')


#def boundary_residual_p_grad(p, x,y,c):
#    return torch.abs(p)
#boundary_cond_p_up = tp.conditions.PINNCondition_CNN(model, bound_sampler_up, boundary_residual_p_grad, dist_matrix=dist, weight=1,name='bound_p_up')



Periodic_sampler=tp.samplers.RandomUniformSampler(Y_interval*C_interval,n_points=250).make_static(resample_interval=2000)#,filter_fn=Inner_filter)


def periodic_residual(u_left,u_right,v_left,v_right,urms_left,urms_right,vrms_left,vrms_right,uv_left,uv_right,p_left,p_right):
    Periodic_condition= torch.abs(u_left - u_right)+torch.abs(v_left - v_right)+torch.abs(urms_left - urms_right)+torch.abs(vrms_left - vrms_right)+torch.abs(uv_left - uv_right)+torch.abs(p_left - p_right)
    return Periodic_condition
periodic_cond=tp.conditions.PeriodicCondition_CNN(model,X_interval,periodic_residual,dist_matrix=dist,non_periodic_sampler=Periodic_sampler, weight=1,name='periodic')


#def periodic_residual_y(v_left,v_right):
#    Periodic_condition= v_left - v_right
#    return Periodic_condition
#periodic_cond_y=tp.conditions.PeriodicCondition_CNN(model,X_interval,periodic_residual_y,dist_matrix=dist,non_periodic_sampler=Periodic_sampler, weight=1,name='periodic_y')

#def periodic_residual_uu(urms_left,urms_right):
#    Periodic_condition= urms_left - urms_right
#    return Periodic_condition
#periodic_cond_uu=tp.conditions.PeriodicCondition_CNN(model,X_interval,periodic_residual_uu,dist_matrix=dist,non_periodic_sampler=Periodic_sampler, weight=1,name='periodic_uu')


#def periodic_residual_vv(vrms_left,vrms_right):
#    Periodic_condition= vrms_left - vrms_right
#    return Periodic_condition
#periodic_cond_vv=tp.conditions.PeriodicCondition_CNN(model,X_interval,periodic_residual_vv,dist_matrix=dist,non_periodic_sampler=Periodic_sampler, weight=1,name='periodic_vv')


#def periodic_residual_uv(uv_left,uv_right):
#    Periodic_condition= uv_left - uv_right
#    return Periodic_condition
#periodic_cond_uv=tp.conditions.PeriodicCondition_CNN(model,X_interval,periodic_residual_uv,dist_matrix=dist,non_periodic_sampler=Periodic_sampler, weight=1,name='periodic_uv')


#def periodic_residual_p(p_left,p_right):
#    Periodic_condition= p_left-p_right
#    return Periodic_condition
#periodic_cond_p=tp.conditions.PeriodicCondition(model,X_interval,periodic_residual_p,non_periodic_sampler=Periodic_sampler, #weight=1,name='periodic_p')
#bound_sampler_left = tp.samplers.RandomUniformSampler(X_interval.boundary_left*Y_interval, n_points=250)
#def boundary_residual_p(p, x,y):
#    return p

#periodic_cond_p=tp.conditions.PeriodicCondition_CNN(model,X_interval,periodic_residual_p,dist_matrix=dist,non_periodic_sampler=Periodic_sampler, weight=1,name='periodic_p')

bound_sampler_left = tp.samplers.RandomUniformSampler(X_interval.boundary_left*Y_interval*C_interval, n_points=250)
def boundary_residual_p(p, x,y):
    return p

boundary_cond_p = tp.conditions.PINNCondition_CNN(model, bound_sampler_left, boundary_residual_p, dist_matrix=dist, weight=1,name='bound_p')

from random import sample
class Data_set_pinn(torch.utils.data.Dataset):
    def __init__(self,Data_Pinn,matrix_mask_Data,epoch_batch_size=1000):
        #self.x_train=data_DF[x_list].values
        #self.Data_Pinn=DF.to_numpy().reshape((1,-1))
        #self.y_train=data_DF[y_list].values
        self.Data_Pinn=torch.cat((Data_Pinn,matrix_mask_Data),1)
        self.N_epochs=epoch_batch_size
        self.length=epoch_batch_size
        #self.y_train=torch.tensor(y_train,dtype=torch.float32)
        #self.x_train=tp.spaces.Points(self.x_train,X*Y)
        #self.y_train=tp.spaces.Points(self.y_train,U*V)
    def __len__(self):
        return self.length
        #return 5

    def __getitem__(self,idx):
        #return self.x_train[idx] , self.y_train[idx]
        #print(torch.as_tensor([self.x_train[idx]]).shape)
        idx_random=sample(range(Data_Pinn.shape[0]),1)
        P_xtrain=self.Data_Pinn[idx_random[0]]
        P_ytrain=self.Data_Pinn[idx_random[0]]
        return P_xtrain,P_ytrain
dataset_turbulent=Data_set_pinn(Data_Pinn,matrix_mask_Data,epoch_batch_size=N_epochs)
Disc_dataloader=DataLoader(dataset_turbulent,batch_size=N_dists,shuffle=True,drop_last=True)

##Learning rate scheduling To-Do -- launch LR scheduling only after first training phase
optim_G = tp.OptimizerSetting(torch.optim.Adam, lr=0.0001,scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,scheduler_args={"patience":10000,"factor":0.8,"verbose":True,"min_lr":0.000005},monitor_lr="train/model_loss")
optim_D = tp.OptimizerSetting(torch.optim.Adam, lr=0.0001,scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,scheduler_args={"patience":10000,"factor":0.8,"verbose":True,"min_lr":0.000005},monitor_lr="train/D_loss")
#solver = tp.solver.Solver([pde_cond_IBM,pde_cond_mass,boundary_cond_x, pde_cond_x,periodic_cond_x,boundary_cond_y, pde_cond_y,periodic_cond_y], optimizer_setting=optim)
##loss terms scheduling
list_of_Losses=[          pde_cond_IBM_low,
                           #pde_cond_IBM_low_uu,
                           #boundary_cond_x,#1000
                           #boundary_cond_y,
                           #boundary_cond_uu,#1000
                           #boundary_cond_vv,
                           #boundary_cond_uv,
                           #boundary_cond_p,
                           periodic_cond,#2000
                           #periodic_cond_y,
                           #periodic_cond_uu,#2000
                           #periodic_cond_vv,
                           #periodic_cond_uv,
                           #periodic_cond_p,
                           boundary_cond_up,
                           #boundary_cond_y_up,
                           #boundary_cond_Re_up,
                           #boundary_cond_uv_up,
                           #boundary_cond_p_up,
                           pde_cond_x,#5000
                           pde_cond_y,#5000
                           pde_cond_mass]
solver = tp.solver.PIAN_Solver_CNN_Wasserstein_LowMem_half(list_of_Losses,#1000
                               generator=model,
                               discriminator=disc,
                          optimizer_setting_G=optim_G,
                        optimizer_setting_D=optim_D,
                         loss_function_schedule=[     
                             {
                        "conditions":list(range(len(list_of_Losses))),
                        "max_iter":200000
                    }
                ],
                          weight_tunning=True,
                          weight_tunning_parameters={
                                    "alfa":0.99,
                                    "E_rho":0.99,
                                    "Temperature":0.1,
                                    "tunning_every_n_steps":100
                          }, ## Default weight-tunning settings
                                GAN_weight=GAN_weight,
                               co_sys=X*Y,
                                   disc_space=U*V*URMS*VRMS*UV,
                                   N_dist=N_dists,## number of 1d roughness -> N 2d flow fileds
                                   L_x=L_x,
                                   N_x=N_x,
                                   N_y=N_y,
                                   dist_repository=repo_dist[:,None,:],
                                   dist_repository_low=dist_new[:,None,:],
                                gpu=GPU,
                                N_x_sub=N_x_sub,
                                dataset_CNN=dataset_turbulent,
                                Grid_data=Grid_data,
                                max_batch=2,
                         )

a,_=next(iter(Disc_dataloader))
a.shape
plt.imshow(a[0,0,0:N_x_sub,:].detach().cpu())
plt.colorbar()
plt.savefig(f"Figs/PIAN_Lowmem/U_snapshot_Real.png")
plt.close()
plt.imshow(a[0,2,0:N_x_sub,:].detach().cpu())
plt.colorbar()
plt.savefig(f"Figs/PIAN_Lowmem/uu_snapshot_Real.png")
plt.close()
plt.imshow(a[0,4,0:N_x_sub,:].detach().cpu())
plt.colorbar()
plt.savefig(f"Figs/PIAN_Lowmem/uv_snapshot_Real.png")
plt.close()
torch.set_float32_matmul_precision('medium')
comet_logger = pl_loggers.CSVLogger(save_dir="logs/")
print(model)
print(disc)
print()
trainer = pl.Trainer(gpus=1,# use one GPU
                     max_steps=160000, # iteration number
                     benchmark=True, # faster if input batch has constant size
                     logger=comet_logger, # for writting into tensorboard
                     log_every_n_steps=100,
                     enable_checkpointing=False,
                     limit_val_batches=0,
                        num_sanity_val_steps=0,
                    reload_dataloaders_every_n_epochs =1) # saving checkpoints ToDo: turn on checkpointing after first training phase
trainer.fit(solver)#,train_dataloaders=Disc_dataloader)


torch.save(model,"Flat_PIAN_CNN_OPT_Wasserstein_gridmesh.pt")








