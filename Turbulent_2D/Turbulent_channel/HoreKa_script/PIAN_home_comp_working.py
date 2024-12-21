 
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

DF_Data=pd.read_csv("Data/Flow_3d_mesh_6_X720Y226_half.csv")
#len(DF_Data[(DF_Data.c==0)&(DF_Data.y<0.4)]["y"].unique())

# GAN macros
GPU="cuda:0"
GAN_weight=1 # Generator weight
L_x=3.6 # length of domain
N_x=240 # grid in x
N_x_sub=70
N_y_F=76 # grid in y
y_lim=0.50
N_y_sub=len(DF_Data[(DF_Data.c==0)&(DF_Data.y<y_lim)]["y"].unique())
N_dists=1 # N 1d roughness for fake images during trainingN_epochs
N_epochs=1000 # N training iteration each epoch
GP_weight=10
Grid_data=False

DF_DIST=pd.read_csv("Data/dist_single.csv",header=None)
x_corr=DF_DIST.iloc[0].to_numpy()
dist_repo=DF_DIST.iloc[1:].to_numpy()
DF_DIST=pd.read_csv("Data/dist_odd.csv",header=None)
dist_repo=np.vstack((dist_repo,DF_DIST.iloc[1:].to_numpy()))
x_corr_new=np.linspace(min(x_corr),max(x_corr),N_x)
dist_repo_low=np.zeros((len(dist_repo),N_x))
for i in range(len(dist_repo)):
    dist_repo_low[i,:]=np.interp(x_corr_new,x_corr,dist_repo[i])
x_corr_new=torch.tensor(x_corr_new,requires_grad=False, dtype=torch.float32,device=GPU)
x_corr=torch.tensor(x_corr,requires_grad=False, dtype=torch.float32,device=GPU)
dist_repo_low=torch.tensor(dist_repo_low,requires_grad=False, dtype=torch.float32,device=GPU)
dist_repo=torch.tensor(dist_repo, requires_grad=False,dtype=torch.float32,device=GPU)
N_c_train=len(dist_repo)

X_interval = tp.domains.Interval(X, 0, L_x) # <-add the bounds of the Interval (0, 2)
Y_interval = tp.domains.Interval(Y, 0, 1.0)
C_interval = tp.domains.Interval(C,0,N_c_train) # number of available roughness profiles

Sim_domain = X_interval*Y_interval*C_interval


Y_interval_sub_low=tp.domains.Interval(Y, 0, 0.15)
Sim_domain_sub_low=X_interval*Y_interval_sub_low*C_interval

Y_interval_sub_log=tp.domains.Interval(Y,y_lim,1)
Sim_domain_sub_log=X_interval*Y_interval_sub_log*C_interval
#Y_interval_sub_up=tp.domains.Interval(Y, 2-0.15, 2)
#Sim_domain_sub_up=X_interval*Y_interval_sub_up*C_interval



#DF_Data=pd.read_csv("Data/Flow_3d_2289_6_Cx3_Cy3X240Y76_half.csv")
#DF=DF_Data[["U","V","urms","vrms","uv",]]

#if len(DF_Data["c"].unique())!=N_c:
#    assert "mismatched case number"# number of available training data
N_c=len(DF_Data["c"].unique()) # number of available training data
DF_vincinity=DF_Data[(DF_Data.y<y_lim)]


#Data_Pinn=torch.tensor(DF_Data[["U","V","urms","vrms","uv"]].to_numpy().transpose((1,0)).reshape((5,N_c,N_x,N_y)),dtype=torch.float32,device=GPU)
Data_Pinn=torch.permute(torch.tensor(DF_vincinity[["U","V","urms","vrms","uv"]].to_numpy(),dtype=torch.float32,device=GPU).reshape((N_c,N_x,N_y_sub,5)),(0,3,1,2))



#eta=torch.cos(torch.pi*(torch.tensor([i for i in range(N_y*2)],requires_grad=False,device=GPU))/(N_y*2-1)) 

ys=DF_vincinity["y"].unique()[:N_y_sub]
ys=torch.tensor(ys,dtype=torch.float32,device=GPU)

DF_DIST=pd.read_csv("Data/dist_train.csv",header=None)
x_corr=DF_DIST.iloc[0].to_numpy()
dist=DF_DIST.iloc[1:].to_numpy()
x_corr_new=np.linspace(min(x_corr),max(x_corr),N_x)
dist_new=np.zeros((len(dist),N_x))
for i in range(len(dist)):
    dist_new[i,:]=np.interp(x_corr_new,x_corr,dist[i])
x_corr_new=torch.tensor(x_corr_new,requires_grad=False, dtype=torch.float32,device=GPU)
x_corr=torch.tensor(x_corr,requires_grad=False, dtype=torch.float32,device=GPU)
dist_new=torch.tensor(dist_new,requires_grad=False, dtype=torch.float32,device=GPU)
#dist=torch.tensor(dist, requires_grad=False,dtype=torch.float32,device=GPU)


ymesh=ys.reshape((1,1,1,-1)).expand((N_c,1,N_x,-1))

matrix_mask_Data=(ymesh<dist_new.reshape((N_c,1,N_x,-1)))|(ymesh>=(2-dist_new.reshape((N_c,1,N_x,-1))))
matrix_mask_Data=matrix_mask_Data.long()


def IBM_irr_filter_low(x,y,c):
    position_r=torch.tensor([sum(x_corr<x_i)-1 for x_i in x],device=GPU)
    #position_r=sum(x_corr<x)
    position_l=position_r-1
    x_new=torch.transpose(x,1,0)[0]
    c_new=torch.transpose(c,1,0)[0]
    c_new = c_new.to(torch.long)
    height=dist_repo[c_new,position_l]+(dist_repo[c_new,position_r]-dist_repo[c_new,position_l])*(x_new-x_corr[position_l])/(x_corr[position_r]-x_corr[position_l])
    return (y[...,0]<=height)
IBM_sampler_irr_low = tp.samplers.RandomUniformSampler(Sim_domain_sub_low,n_points=1000,filter_fn=IBM_irr_filter_low).make_static(resample_interval=1000)






def Inner_filter(x,y,c):
    position_r=torch.tensor([sum(x_corr<x_i)-1 for x_i in x],device=GPU)
    #position_r=sum(x_corr<x)
    position_l=position_r-1
    x_new=torch.transpose(x,1,0)[0]
    c_new=torch.transpose(c,1,0)[0]
    c_new = c_new.to(torch.long)
    height=dist_repo[c_new,position_l]+(dist_repo[c_new,position_r]-dist_repo[c_new,position_l])*(x_new-x_corr[position_l])/(x_corr[position_r]-x_corr[position_l])
    return (y[...,0]>height)
inner_sampler = tp.samplers.RandomUniformSampler(Sim_domain, n_points=1000,filter_fn=Inner_filter).make_static(resample_interval=200)#,filter_fn=Inner_filter)





def Log_filter(x,y,c):
    return (y[...,0]>y_lim)
log_sampler = tp.samplers.RandomUniformSampler(Sim_domain_sub_log, n_points=3000,filter_fn=Log_filter)#.make_static(resample_interval=200)#,filter_fn=Inner_filter)



bound_sampler_up = tp.samplers.RandomUniformSampler(X_interval*Y_interval.boundary_right*C_interval, n_points=500)





def self_sin(input):
    return input.sin()
def self_cos(input):
    return input.cos()


import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(seed=42)

class FCN_model_Fourier_Feature_CNN(nn.Module):
    def __init__(self,input_space,output_space,N_features,sigma_1=1,sigma_2=15):
        super().__init__()
        self.W_1 = torch.tensor(torch.randn(input_space.dim , N_features //2, dtype=torch.float32,device=GPU) * sigma_1, dtype=torch.float32, requires_grad=False)
        self.W_2 = torch.tensor(torch.randn(input_space.dim , N_features //2, dtype=torch.float32,device=GPU) * sigma_2, dtype=torch.float32, requires_grad=False)
        self.register_buffer("selfW1", self.W_1, persistent=False)
        self.register_buffer("selfW2", self.W_2, persistent=False)
        self.output_space=output_space
        self.fc1_l=nn.Linear(in_features=N_features,out_features=150,device=GPU)
        self.fc2_l=nn.Linear(in_features=150,out_features=150,device=GPU)
        self.fc3_l=nn.Linear(in_features=150,out_features=150,device=GPU)
        ###
        self.fc1_r=nn.Linear(in_features=N_features,out_features=150,device=GPU)
        self.fc2_r=nn.Linear(in_features=150,out_features=150,device=GPU)
        self.fc3_r=nn.Linear(in_features=150,out_features=150,device=GPU)


        self.conv1=nn.Conv1d(1,5,3,stride=1,padding=1,device=GPU) #((720-3+2*1)/1)+1 *8=720 *5
        self.act1=nn.LeakyReLU(0.1)
        #self.pool1=nn.MaxPool1d(kernel_size=2) # stride=2   (720-2)/2 + 1 *8 = 360 *3

        self.conv2=nn.Conv1d(5,9,3,stride=1,padding=1,device=GPU) # ((720-3+2*1)/1)+1 *32 = 720*9
        self.act2=nn.LeakyReLU(0.1)
        #self.pool2=nn.MaxPool1d(kernel_size=2) #360*6

        self.conv3=nn.Conv1d(9,15,5,stride=3,padding=1,device=GPU) #((720-5+2*1)/3)+1 *32 = 240*15
        self.act3=nn.LeakyReLU(0.1)
        #self.pool3=nn.MaxPool1d(kernel_size=2)# 120*9
        self.conv4=nn.Conv1d(15,20,5,stride=3,padding=1,device=GPU) #((120-5+2*1)/3)+1 *32 = 40*20
        self.act4=nn.LeakyReLU(0.1)
        #self.pool4=nn.MaxPool1d(kernel_size=2)# 20*20

        self.flat=nn.Flatten()

        self.fc_combo1=nn.Linear(in_features=1600+300,out_features=1000,device=GPU)
        self.fc_combo2=nn.Linear(in_features=1000,out_features=1000,device=GPU)
        self.fc_combo3=nn.Linear(in_features=1000,out_features=1000,device=GPU)
        self.fc_combo4=nn.Linear(in_features=1000,out_features=1000,device=GPU)
        self.fc_combo5=nn.Linear(in_features=1000,out_features=1000,device=GPU)
        self.fc_combo6=nn.Linear(in_features=1000,out_features=500,device=GPU)



        self.out=nn.Linear(in_features=500,out_features=output_space.dim,device=GPU)
    def forward(self,dist_1d,t):
        t=t.as_tensor[:,0:2]
        t_1=torch.concat((self_sin(torch.matmul(t,self.W_1)),self_cos(torch.matmul(t,self.W_1))),1)
        t_1=self_sin(self.fc1_l(t_1))
        t_1=self_sin(self.fc2_l(t_1))
        t_1=self_sin(self.fc3_l(t_1))
        t_2=torch.concat((self_sin(torch.matmul(t,self.W_2)),self_cos(torch.matmul(t,self.W_2))),1)
        t_2=self_sin(self.fc1_r(t_2))
        t_2=self_sin(self.fc2_r(t_2))
        t_2=self_sin(self.fc3_r(t_2))
        t_figure=self.act1(self.conv1(dist_1d))
        t_figure=self.act2(self.conv2(t_figure))
        t_figure=self.act3(self.conv3(t_figure))
        t_figure=self.act4(self.conv4(t_figure))
        t=torch.concat((t_1,t_2,self.flat(t_figure)),1)
        t=self_sin(self.fc_combo1(t))
        t=self_sin(self.fc_combo2(t))
        t=self_sin(self.fc_combo3(t))
        t=self_sin(self.fc_combo4(t))
        t=self_sin(self.fc_combo5(t))
        t=self_sin(self.fc_combo6(t))
        t=self.out(t)
        return tp.problem.spaces.Points(t, self.output_space)

#model=FCN_model_Fourier_Feature_CNN(input_space=X*Y,output_space=U*V*URMS*VRMS*UV*P,N_features=300)
model=torch.load("Model_home_complex_5.pt")
model.train()


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
        self.fc1 = nn.Linear(3072, 4096)
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
#disc=ResNet(ResidualBlock, [2, 2, 2, 2]).to(GPU)
disc=torch.load("Critic_log_sum_logarithmic_80_50_new.pt")
disc.train()


def pde_IBM(u,v):
    return torch.sqrt(torch.square(u)+torch.square(v))
pde_cond_IBM_low = tp.conditions.PINNCondition_CNN(model, IBM_sampler_irr_low, pde_IBM, dist_matrix=dist_repo,weight=100,name='IBM_low')


def log_diagnostic(u,x,y,c):
    c_new=torch.transpose(c,1,0)[0]
    c_new = c_new.to(torch.long)
    utau=800/((1-torch.mean(dist_repo[c_new,:],dim=1)[:,None])*6000)
    # print(x,y,u,tp.utils.grad(u,y),tp.utils.grad(u,y)/(800*utau)*(y-torch.mean(dist_repo[c_new,:],dim=1)[:,None])*800-(2.63))
    return torch.abs(tp.utils.grad(u,y)/(800*utau)*(y-torch.mean(dist_repo[c_new,:],dim=1)[:,None])*800-(2.6))
pde_cond_log=tp.conditions.PINNCondition_CNN(model, log_sampler, log_diagnostic, dist_matrix=dist_repo,weight=50,name='logarithmic')




def pde_mass(u,v,x,y):
    return tp.utils.grad(u,x)+tp.utils.grad(v,y)-0.0
pde_cond_mass = tp.conditions.PINNCondition_CNN(model, inner_sampler,pde_mass, dist_matrix=dist_repo, weight=50,name='Conti')



def pde_residual_x(u,v, x, y,urms,vrms,uv,p,c):
    c_new=torch.transpose(c,1,0)[0]
    c_new = c_new.to(torch.long)
    residual_momentum= u*tp.utils.grad(u,x)+ v*tp.utils.grad(u,y)+(-tp.utils.laplacian(u, x)-tp.utils.laplacian(u,y))/6000 - ( (2/15)**2/((1-torch.mean(dist_repo[c_new,:],dim=1)[:,None])**3) ) +tp.utils.grad(torch.square(urms/6),x)+tp.utils.grad(uv/60,y)+tp.utils.grad(p,x)
    return residual_momentum

pde_cond_x = tp.conditions.PINNCondition_CNN(model, inner_sampler, pde_residual_x, dist_matrix=dist_repo,weight=51,name='Momentum_x')



def pde_residual_y(u,v, x, y,urms,vrms,uv,p):
    residual_momentum= u*tp.utils.grad(v,x)+v*tp.utils.grad(v,y)+(-tp.utils.laplacian(v, x)-tp.utils.laplacian(v,y))/6000+tp.utils.grad(uv/60,x)+tp.utils.grad(torch.square(vrms/6),y)+tp.utils.grad(p,y)
    return residual_momentum

pde_cond_y = tp.conditions.PINNCondition_CNN(model, inner_sampler, pde_residual_y, dist_matrix=dist_repo,weight=50,name='Momentum_y')


def boundary_residual(u,v,urms,vrms,uv,p, x,y,c):
    #un=torch.abs(tp.utils.grad(u,y))+torch.abs(v)+torch.abs(tp.utils.grad(urms,y))+torch.abs(tp.utils.grad(vrms,y))+torch.abs(uv)+torch.abs(p)
    un=torch.abs(v)+torch.abs(tp.utils.grad(urms,y))+torch.abs(tp.utils.grad(vrms,y))+torch.abs(uv)+torch.abs(p)
 
    return un
boundary_cond_up = tp.conditions.PINNCondition_CNN(model, bound_sampler_up, boundary_residual, dist_matrix=dist_repo, weight=1,name='bound_up')


Periodic_sampler=tp.samplers.RandomUniformSampler(Y_interval*C_interval,n_points=250).make_static(resample_interval=2000)#,filter_fn=Inner_filter)


def periodic_residual_x(u_left,u_right):
    Periodic_condition= u_left - u_right
    return Periodic_condition
periodic_cond_x=tp.conditions.PeriodicCondition_CNN(model,X_interval,periodic_residual_x,dist_matrix=dist_repo,non_periodic_sampler=Periodic_sampler, weight=1,name='periodic_x')


def periodic_residual(u_left,u_right,v_left,v_right,urms_left,urms_right,vrms_left,vrms_right,uv_left,uv_right,p_left,p_right):
    Periodic_condition= torch.abs(u_left - u_right)+torch.abs(v_left - v_right)+torch.abs(urms_left - urms_right)+torch.abs(vrms_left - vrms_right)+torch.abs(uv_left - uv_right)+torch.abs(p_left - p_right)
    return Periodic_condition
periodic_cond=tp.conditions.PeriodicCondition_CNN(model,X_interval,periodic_residual,dist_matrix=dist_repo,non_periodic_sampler=Periodic_sampler, weight=1,name='periodic')

bound_sampler_left = tp.samplers.RandomUniformSampler(X_interval.boundary_left*Y_interval*C_interval, n_points=250)
def boundary_residual_p(p, x,y):
    return p

boundary_cond_p = tp.conditions.PINNCondition_CNN(model, bound_sampler_left, boundary_residual_p, dist_matrix=dist_repo, weight=1,name='bound_p')


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
list_of_Losses=[         pde_cond_log,
                pde_cond_IBM_low,
                           periodic_cond,#2000
                           boundary_cond_up,
                           pde_cond_x,#5000
                           pde_cond_y,#5000
                           pde_cond_mass]
solver = tp.solver.PIAN_Solver_CNN_Wasserstein_LowMem_half_logarithmic(list_of_Losses,#1000
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
                                    ys=ys,
                                   N_x=N_x,
                                   N_y=N_y_sub,
                                   dist_repository=dist_repo[:,None,:],
                                   dist_repository_low=dist_repo_low[:,None,:],
                                gpu=GPU,
                                N_x_sub=N_x_sub,
                                dataset_CNN=dataset_turbulent,
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
trainer = pl.Trainer(accelerator='gpu',# use one GPU
                     devices=[0,1,2,3],
                     max_steps=200000, # iteration number
                     benchmark=True, # faster if input batch has constant size
                     logger=comet_logger, # for writting into tensorboard
                     log_every_n_steps=100,
                     enable_checkpointing=False,
                     limit_val_batches=0,
                        num_sanity_val_steps=0,
                    reload_dataloaders_every_n_epochs =1) # saving checkpoints ToDo: turn on checkpointing after first training phase
trainer.fit(solver)#,train_dataloaders=Disc_dataloader)


torch.save(model,"Model_home_complex_6.pt")
torch.save(disc,"Critic_log_sum_logarithmic_80_50_new.pt")
#torch.save({"G_state_dict":model.state_dict(),
#           "D_state_dict":disc.state_dict()},
#          "Models_PIAN_log_sum_logarithimic_80*50.pt")








