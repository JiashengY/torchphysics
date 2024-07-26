from typing import Dict
import warnings
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .problem.spaces.points import Points
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from random import sample
import time
class OptimizerSetting:
    """
    A helper class to sum up the optimization setup in a single class.
    """
    def __init__(self, optimizer_class, lr, optimizer_args={}, scheduler_class=None,
                 scheduler_args={}, scheduler_frequency=1,monitor_lr=None):
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.optimizer_args = optimizer_args
        self.scheduler_class = scheduler_class
        self.scheduler_args = scheduler_args
        self.scheduler_frequency = scheduler_frequency
        self.monitor_lr=monitor_lr

class Solver(pl.LightningModule):
    """
    A LightningModule that handles optimization and metric logging of given
    conditions.

    Parameters
    ----------
    train_conditions : tuple or list
        Tuple or list of conditions to be optimized. The weighted sum of their
        losses will be computed and minimized.
    val_conditions : tuple or list
        Conditions to be tracked during the validation part of the training, can
        be used e.g. to track errors comparede to measured data.
    optimizer_setting : OptimizerSetting
        A OptimizerSetting object that contains all necessary parameters for
        optimizing, see :class:`OptimizerSetting`.
    """
    def __init__(self,
                 train_conditions,
                 val_conditions=(),
                 optimizer_setting=OptimizerSetting(torch.optim.Adam,
                                                    1e-3),
                 loss_function_schedule=[{ ##############Modified JY#####################
                        "conditions":[],
                        "max_iter":-1
                    }
                ],
                 weight_tunning=True,
                 weight_tunning_parameters={
                     "alfa":0.99,
                     "E_rho":0.99,
                     "Temperature":1,
                     "tunning_every_n_steps":100
                 }):######################################################################
        super().__init__()
        self.train_conditions = nn.ModuleList(train_conditions)
        self.val_conditions = nn.ModuleList(val_conditions)
        self.optimizer_setting = optimizer_setting
        ############################## Modified JY ######################################
        self.loss_function_schedule=loss_function_schedule
        self.weight_tunning=weight_tunning
        if self.weight_tunning:
            self.alfa=weight_tunning_parameters["alfa"]
            self.E_rho=weight_tunning_parameters["E_rho"]
            self.Temperature=weight_tunning_parameters["Temperature"]
            self.nsteps=weight_tunning_parameters["tunning_every_n_steps"]
        else:
            self.nsteps=0
        ###################################################################################
    def train_dataloader(self):
        """"""
        # HACK: create an empty trivial dataloader, since real data is loaded
        # in conditions
        steps = self.trainer.max_steps
        if steps is None:
            warnings.warn("The maximum amount of iterations should be defined in"
                "trainer.max_steps. If undefined, the solver will train in epochs"
                "of 1000 steps.")
            steps = 1000
        return torch.utils.data.DataLoader(torch.empty(steps))

    def val_dataloader(self):
        """"""
        # HACK: we perform only a single step during validation,
        return torch.utils.data.DataLoader(torch.empty(1))

    def _set_lr_scheduler(self, optimizer):
        lr_scheduler = self.scheduler['class'](optimizer, **self.scheduler['args'])
        lr_scheduler = {'scheduler': lr_scheduler, 'name': 'learning_rate',
                        'interval': 'epoch', 'frequency': 1}
        for input_name in self.scheduler:
            if not input_name in ['class', 'args']:
                lr_scheduler[input_name] = self.scheduler[input_name]
        return lr_scheduler

    def on_train_start(self):
        # move static data to correct device:
        for condition in self.train_conditions:
            condition._move_static_data(self.device)
        for condition in self.val_conditions:
            condition._move_static_data(self.device)
        self.n_training_step = 0

####################Modified JY######################################
###Multi-Objective Loss Balancing for Physics-Informed Deep Learning https://doi.org/10.13140/rg.2.2.20057.24169
    def _ReLoBRALO(self,list_cond_loss,train_conditions_index):   ## RElative LOss Balancing with RAndom LOokback
        m=len(list_cond_loss)
        max_bal=max([list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i]) for i in range(m)])##very large number -- preventing softmax overflow
        sum_exp_L=sum([torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i])-max_bal) for i in range(m)])
        lambda_bal=[self.train_conditions[train_conditions_index[i]].base_weight*m*torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i])-max_bal) / sum_exp_L for i in range(m)]
        max_init=max([list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i]) for i in range(m)])
        sum_exp_L_init=sum([torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i])-max_init) for i in range(m)])
        lambda_bal_init=[self.train_conditions[train_conditions_index[i]].base_weight*m*torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i])-max_init) / sum_exp_L_init for i in range(m)]
        if self.E_rho==1:
            rho=torch.tensor(1)
        else:
            rho=torch.bernoulli(torch.tensor(self.E_rho)) #### all terms share bernoulli random number
        #print(sum_exp_L.item(),sum_exp_L_init.item())
        for i in range(m):
            #print(self.train_conditions[train_conditions_index[i]].name)
            #print(alfa*(rho*self.train_conditions[i].weight+(1-rho)*lambda_bal_init[i]).item(),(1-alfa)*(lambda_bal[i]).item())
            self.train_conditions[train_conditions_index[i]].weight=(self.alfa*(rho*self.train_conditions[train_conditions_index[i]].base_weight+(1-rho)*lambda_bal_init[i])+(1-self.alfa)*lambda_bal[i]).item()
            self.log(f'weight/{self.train_conditions[train_conditions_index[i]].name}', self.train_conditions[train_conditions_index[i]].weight)
        
    def _baseweight_tunner(self,list_cond_loss,train_conditions_index):
        m=len(list_cond_loss)
        for i in range(m):
            factor=torch.bernoulli(torch.tensor(0.2)).item()
            if 1/list_cond_loss[i] >1:
                self.train_conditions[train_conditions_index[i]].base_weight=1/list_cond_loss[i]
            #print(i,self.train_conditions[train_conditions_index[i]].base_weight,factor)
                self.train_conditions[train_conditions_index[i]].base_weight=self.train_conditions[train_conditions_index[i]].base_weight*(10**factor)
            else:
                self.train_conditions[train_conditions_index[i]].base_weight=10**factor
            #print(i,self.train_conditions[train_conditions_index[i]].base_weight)


    def training_step(self, batch, batch_idx): ####################### modified JY ################################
        loss = torch.zeros(1, requires_grad=True, device=self.device)
        ######### first set of loss functions #######
        if self.n_training_step<=self.loss_function_schedule[0]["max_iter"]:   
            train_conditions_index=self.loss_function_schedule[0]["conditions"]
            if self.n_training_step==0:
                self.list_cond_loss_his_init=[]
                self.list_cond_loss_his=[]
                for i in self.train_conditions:
                    i.base_weight=i.weight
                    #i.weight=1
                for condition in [self.train_conditions[j] for j in train_conditions_index]:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.base_weight*cond_loss
                    #self.train_conditions[i].weight=1
                    self.list_cond_loss_his_init.append(cond_loss)
                    self.list_cond_loss_his=self.list_cond_loss_his_init
                    list_cond_loss=self.list_cond_loss_his_init
            else:
                list_cond_loss=[]
                for condition in [self.train_conditions[j] for j in train_conditions_index]:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.weight*cond_loss
                    list_cond_loss.append(cond_loss)
                if self.weight_tunning & (self.n_training_step%self.nsteps==0):
                    self._ReLoBRALO(list_cond_loss,train_conditions_index)
            self.list_cond_loss_his=list_cond_loss
            self.log('train/loss', loss)
            self.n_training_step += 1
            #if self.n_training_step%1000==0:
                #self._baseweight_tunner()
            return loss
        
    
        if self.n_training_step>=self.loss_function_schedule[-1]["max_iter"]:
            train_conditions_index=list(range(len(self.train_conditions)))
            n_step_init=self.loss_function_schedule[-1]["max_iter"]+1
            if self.n_training_step<=(self.loss_function_schedule[-1]["max_iter"]+42):
                self.list_cond_loss_his_init=[]
                self.list_cond_loss_his=[]
                for condition in self.train_conditions:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.base_weight*cond_loss
                    self.list_cond_loss_his_init.append(cond_loss)
                    self.list_cond_loss_his=self.list_cond_loss_his_init
                    list_cond_loss=self.list_cond_loss_his_init
            else:
                list_cond_loss=[]
                for condition in self.train_conditions:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.weight*cond_loss
                    list_cond_loss.append(cond_loss)
                #if self.weight_tunning & ((self.n_training_step-n_step_init)%(self.nsteps*10))==0:
                #    self._baseweight_tunner(list_cond_loss,train_conditions_index)
                if self.weight_tunning &((self.n_training_step-n_step_init)%self.nsteps==0):
                    self._ReLoBRALO(list_cond_loss,train_conditions_index)
            self.list_cond_loss_his=list_cond_loss
            self.log('train/loss', loss)
            self.n_training_step += 1
            
            return loss
        

        for i in range(len(self.loss_function_schedule)-1):
            if (self.n_training_step<=self.loss_function_schedule[i+1]["max_iter"]) & (self.n_training_step>self.loss_function_schedule[i]["max_iter"]):
                train_conditions_index=self.loss_function_schedule[i+1]["conditions"]
                n_step_init=self.loss_function_schedule[i]["max_iter"]+1
                break
        if self.n_training_step < (n_step_init+42):### Running-in buffer
            self.list_cond_loss_his_init=[]
            self.list_cond_loss_his=[]
            for condition in [self.train_conditions[j] for j in train_conditions_index]:
                cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                self.log(f'train/{condition.name}', cond_loss)
                loss = loss + condition.base_weight*cond_loss
                self.list_cond_loss_his_init.append(cond_loss)
                self.list_cond_loss_his=self.list_cond_loss_his_init
                list_cond_loss=self.list_cond_loss_his_init
        else:
            list_cond_loss=[]
            for condition in [self.train_conditions[j] for j in train_conditions_index]:
                cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                self.log(f'train/{condition.name}', cond_loss)
                loss = loss + condition.weight*cond_loss
                list_cond_loss.append(cond_loss)
            #if self.weight_tunning & ((n_step_init-self.n_training_step)%(self.nsteps*10)==0):
            #    self._baseweight_tunner(list_cond_loss,train_conditions_index)
            if self.weight_tunning & ((n_step_init-self.n_training_step)%self.nsteps==0):
                    self._ReLoBRALO(list_cond_loss,train_conditions_index)
        self.list_cond_loss_his=list_cond_loss
        self.log('train/loss', loss)
        self.n_training_step += 1
        
        return loss

    def validation_step(self, batch, batch_idx):
        for condition in self.val_conditions:
            torch.set_grad_enabled(condition.track_gradients is not False)
            self.log(f'val/{condition.name}', condition(device=self.device))

    def configure_optimizers(self):
        optimizer = self.optimizer_setting.optimizer_class(
            self.parameters(),
            lr = self.optimizer_setting.lr,
            **self.optimizer_setting.optimizer_args
        )
        if self.optimizer_setting.scheduler_class is None:
            return optimizer

        lr_scheduler = self.optimizer_setting.scheduler_class(optimizer,
            **self.optimizer_setting.scheduler_args
        )
        lr_scheduler = {'scheduler': lr_scheduler, 'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': self.optimizer_setting.scheduler_frequency,
                        'monitor': self.optimizer_setting.monitor_lr}
        for input_name in self.optimizer_setting.scheduler_args:
            lr_scheduler[input_name] = self.optimizer_setting.scheduler_args[input_name]
        return [optimizer], [lr_scheduler]







class PIAN_Solver(pl.LightningModule):
    """
    A LightningModule that handles optimization and metric logging of given
    conditions.

    Parameters
    ----------
    train_conditions : tuple or list
        Tuple or list of conditions to be optimized. The weighted sum of their
        losses will be computed and minimized.
    val_conditions : tuple or list
        Conditions to be tracked during the validation part of the training, can
        be used e.g. to track errors comparede to measured data.
    optimizer_setting : OptimizerSetting
        A OptimizerSetting object that contains all necessary parameters for
        optimizing, see :class:`OptimizerSetting`.
    """
    def __init__(self,
                 train_conditions,
                                  generator,
                 discriminator,
                 co_sys,
                 val_conditions=(),
                 optimizer_setting_G=OptimizerSetting(torch.optim.Adam,
                                                    1e-3),
                 optimizer_setting_D=OptimizerSetting(torch.optim.Adam,
                                                    1e-3),
                 loss_function_schedule=[{ ##############Modified JY#####################
                        "conditions":[],
                        "max_iter":-1
                    }
                ],
                 weight_tunning=True,
                 weight_tunning_parameters={
                     "alfa":0.99,
                     "E_rho":0.99,
                     "Temperature":1,
                     "tunning_every_n_steps":100
                 },
                 GAN_weight=1000,
                 rand_size=20,
                 rand_bound=3.6,
                 N_Profile_points=20,
                 Log_dir="runs/"):######################################################################
        super().__init__()
        self.writer_loss=SummaryWriter(Log_dir)
        self.writer_generator=SummaryWriter(Log_dir)
        self.co_sys=co_sys
        self.GAN_weight=GAN_weight
        self.rand_size=rand_size
        self.generator=generator
        self.discriminator=discriminator
        self.rand_bound=rand_bound
        self.N_points=N_Profile_points
        self.train_conditions = nn.ModuleList(train_conditions)
        self.val_conditions = nn.ModuleList(val_conditions)
        self.optimizer_setting_G = optimizer_setting_G
        self.optimizer_setting_D = optimizer_setting_D
        ############################## Modified JY ######################################
        eta=torch.cos(torch.pi*(torch.tensor([i for i in range(self.N_points)]))/(self.N_points)) 
        self.ys=(1-torch.tensor(eta))
        self.loss_function_schedule=loss_function_schedule
        self.weight_tunning=weight_tunning
        if self.weight_tunning:
            self.alfa=weight_tunning_parameters["alfa"]
            self.E_rho=weight_tunning_parameters["E_rho"]
            self.Temperature=weight_tunning_parameters["Temperature"]
            self.nsteps=weight_tunning_parameters["tunning_every_n_steps"]
        else:
            self.nsteps=0
        ###################################################################################
    def train_dataloader(self):
        """"""
        # HACK: create an empty trivial dataloader, since real data is loaded
        # in conditions
        steps = self.trainer.max_steps
        if steps is None:
            warnings.warn("The maximum amount of iterations should be defined in"
                "trainer.max_steps. If undefined, the solver will train in epochs"
                "of 1000 steps.")
            steps = 1000
        return torch.utils.data.DataLoader(torch.empty(steps))

    def val_dataloader(self):
        """"""
        # HACK: we perform only a single step during validation,
        return torch.utils.data.DataLoader(torch.empty(1))

    def _set_lr_scheduler(self, optimizer):
        lr_scheduler = self.scheduler['class'](optimizer, **self.scheduler['args'])
        lr_scheduler = {'scheduler': lr_scheduler, 'name': 'learning_rate',
                        'interval': 'epoch', 'frequency': 1}
        for input_name in self.scheduler:
            if not input_name in ['class', 'args']:
                lr_scheduler[input_name] = self.scheduler[input_name]
        return lr_scheduler

    def on_train_start(self):
        # move static data to correct device:
        for condition in self.train_conditions:
            condition._move_static_data(self.device)
        for condition in self.val_conditions:
            condition._move_static_data(self.device)
        self.n_training_step = 0

####################Modified JY######################################
###Multi-Objective Loss Balancing for Physics-Informed Deep Learning https://doi.org/10.13140/rg.2.2.20057.24169
    def _ReLoBRALO(self,list_cond_loss,train_conditions_index):   ## RElative LOss Balancing with RAndom LOokback
        m=len(list_cond_loss)
        max_bal=max([list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i]) for i in range(m)])##very large number -- preventing softmax overflow
        sum_exp_L=sum([torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i])-max_bal) for i in range(m)])
        lambda_bal=[self.train_conditions[train_conditions_index[i]].base_weight*m*torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i])-max_bal) / sum_exp_L for i in range(m)]
        max_init=max([list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i]) for i in range(m)])
        sum_exp_L_init=sum([torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i])-max_init) for i in range(m)])
        lambda_bal_init=[self.train_conditions[train_conditions_index[i]].base_weight*m*torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i])-max_init) / sum_exp_L_init for i in range(m)]
        if self.E_rho==1:
            rho=torch.tensor(1)
        else:
            rho=torch.bernoulli(torch.tensor(self.E_rho)) #### all terms share bernoulli random number
        #print(sum_exp_L.item(),sum_exp_L_init.item())
        for i in range(m):
            #print(self.train_conditions[train_conditions_index[i]].name)
            #print(alfa*(rho*self.train_conditions[i].weight+(1-rho)*lambda_bal_init[i]).item(),(1-alfa)*(lambda_bal[i]).item())
            self.train_conditions[train_conditions_index[i]].weight=(self.alfa*(rho*self.train_conditions[train_conditions_index[i]].base_weight+(1-rho)*lambda_bal_init[i])+(1-self.alfa)*lambda_bal[i]).item()
            self.log(f'weight/{self.train_conditions[train_conditions_index[i]].name}', self.train_conditions[train_conditions_index[i]].weight)
        
    def _baseweight_tunner(self,list_cond_loss,train_conditions_index):
        m=len(list_cond_loss)
        for i in range(m):
            factor=torch.bernoulli(torch.tensor(0.2)).item()
            if 1/list_cond_loss[i] >1:
                self.train_conditions[train_conditions_index[i]].base_weight=1/list_cond_loss[i]
            #print(i,self.train_conditions[train_conditions_index[i]].base_weight,factor)
                self.train_conditions[train_conditions_index[i]].base_weight=self.train_conditions[train_conditions_index[i]].base_weight*(10**factor)
            else:
                self.train_conditions[train_conditions_index[i]].base_weight=10**factor
            #print(i,self.train_conditions[train_conditions_index[i]].base_weight)


    def training_step(self, batch, batch_idx,optimizer_idx): ####################### modified JY ################################
        real_profiles,_=batch
        loss = torch.zeros(1, requires_grad=True, device=self.device)
        ######### first set of loss functions #######
        if self.n_training_step<=self.loss_function_schedule[0]["max_iter"]:   
            train_conditions_index=self.loss_function_schedule[0]["conditions"]
            if optimizer_idx==0:
                if self.n_training_step==0:
                    self.list_cond_loss_his_init=[]
                    self.list_cond_loss_his=[]
                    for i in self.train_conditions:
                        i.base_weight=i.weight
                    #i.weight=1
                    for condition in [self.train_conditions[j] for j in train_conditions_index]:
                        cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                        self.log(f'train/{condition.name}', cond_loss)
                        self.writer_loss(f'train/{condition.name}', cond_loss)
                        loss = loss + condition.base_weight*cond_loss
                        #self.train_conditions[i].weight=1
                        self.list_cond_loss_his_init.append(cond_loss)
                        self.list_cond_loss_his=self.list_cond_loss_his_init
                        list_cond_loss=self.list_cond_loss_his_init
                else:
                    list_cond_loss=[]
                    for condition in [self.train_conditions[j] for j in train_conditions_index]:
                        cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                        self.log(f'train/{condition.name}', cond_loss)
                        loss = loss + condition.weight*cond_loss
                        list_cond_loss.append(cond_loss)
                        if self.n_training_step%100<=5:
                            self.writer_loss(f'train/{condition.name}', cond_loss)
                    if self.weight_tunning & (self.n_training_step%self.nsteps==0):
                        self._ReLoBRALO(list_cond_loss,train_conditions_index)
                self.list_cond_loss_his=list_cond_loss
                self.log('train/PINN_loss', loss)
                self.n_training_step += 1
                positions=torch.rand(1,self.rand_size)*self.rand_bound
                #####generator
                fake_profiles=self(positions)  #######20 : random profiles
                y_hat=self.discriminator(fake_profiles)
                y=torch.ones(positions.size(1),1)
                g_loss=self.adversarial_loss(y_hat,y)
                self.log('train/G_loss', g_loss)
                if self.n_training_step%100<=5:
                    self.writer_loss('train/G_loss', g_loss)
                loss=loss+self.GAN_weight*g_loss
                self.log("train/model_loss",loss)
            #if self.n_training_step%1000==0:
                #self._baseweight_tunner()
            elif optimizer_idx==1:
                positions=torch.rand(1,self.rand_size)*self.rand_bound
                y_hat_real=self.discriminator(real_profiles)
                y_real=torch.ones(real_profiles.size(0),1)
                real_loss=self.adversarial_loss(y_hat_real,y_real)
                y_hat_fake=self.discriminator(self(positions).detach())
                y_fake=torch.zeros(positions.size(1),1)
                fake_loss=self.adversarial_loss(y_hat_fake,y_fake)
                d_loss=(real_loss+fake_loss)/2
                self.log('train/D_loss', d_loss)
                if self.n_training_step%100<=5:
                    self.writer_loss('train/D_loss', d_loss)
                loss=loss+self.GAN_weight*d_loss
            return loss
        
    
        if self.n_training_step>=self.loss_function_schedule[-1]["max_iter"]:
            train_conditions_index=list(range(len(self.train_conditions)))
            n_step_init=self.loss_function_schedule[-1]["max_iter"]+1
            if optimizer_idx==0:
                if self.n_training_step<=(self.loss_function_schedule[-1]["max_iter"]+42):
                    self.list_cond_loss_his_init=[]
                    self.list_cond_loss_his=[]
                    for condition in self.train_conditions:
                        cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                        self.log(f'train/{condition.name}', cond_loss)
                        if self.n_training_step%100<=5:
                            self.writer_loss(f'train/{condition.name}', cond_loss)
                        loss = loss + condition.base_weight*cond_loss
                        self.list_cond_loss_his_init.append(cond_loss)
                        self.list_cond_loss_his=self.list_cond_loss_his_init
                        list_cond_loss=self.list_cond_loss_his_init
                else:
                    list_cond_loss=[]
                    for condition in self.train_conditions:
                        cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                        self.log(f'train/{condition.name}', cond_loss)
                        if self.n_training_step%100<=5:
                            self.writer_loss(f'train/{condition.name}', cond_loss)
                        loss = loss + condition.weight*cond_loss
                        list_cond_loss.append(cond_loss)
                #if self.weight_tunning & ((self.n_training_step-n_step_init)%(self.nsteps*10))==0:
                #    self._baseweight_tunner(list_cond_loss,train_conditions_index)
                    if self.weight_tunning&((self.n_training_step-n_step_init)%self.nsteps==0):
                        self._ReLoBRALO(list_cond_loss,train_conditions_index)
                self.list_cond_loss_his=list_cond_loss
                self.log('train/PINN_loss', loss)
                self.n_training_step += 1
            
                positions=torch.rand(1,self.rand_size)*self.rand_bound
                #####generator
                fake_profiles=self(positions)  #######20 : random profiles
                y_hat=self.discriminator(fake_profiles)
                y=torch.ones(positions.size(1),1)
                g_loss=self.adversarial_loss(y_hat,y)
                self.log('train/G_loss', g_loss)
                if self.n_training_step%100<=5:
                    self.writer_loss('train/G_loss', g_loss)
                loss=loss+self.GAN_weight*g_loss
                self.log("train/model_loss",loss)
            #if self.n_training_step%1000==0:
                #self._baseweight_tunner()
            if optimizer_idx==1:
                y_hat_real=self.discriminator(real_profiles)
                y_real=torch.ones(real_profiles.size(0),1)
                real_loss=self.adversarial_loss(y_hat_real,y_real)
                y_hat_fake=self.discriminator(self(positions).detach())
                y_fake=torch.zeros(positions.size(1),1)
                fake_loss=self.adversarial_loss(y_hat_fake,y_fake)
                d_loss=(real_loss+fake_loss)/2
                self.log('train/D_loss', d_loss)
                if self.n_training_step%100<=5:
                    self.writer_loss('train/D_loss', d_loss)
                loss=loss+self.GAN_weight*d_loss
            return loss
        

        for i in range(len(self.loss_function_schedule)-1):
            if (self.n_training_step<=self.loss_function_schedule[i+1]["max_iter"]) & (self.n_training_step>self.loss_function_schedule[i]["max_iter"]):
                train_conditions_index=self.loss_function_schedule[i+1]["conditions"]
                n_step_init=self.loss_function_schedule[i]["max_iter"]+1
                break
        if optimizer_idx==0:
            if self.n_training_step < (n_step_init+42):### Running-in buffer
                self.list_cond_loss_his_init=[]
                self.list_cond_loss_his=[]
                for condition in [self.train_conditions[j] for j in train_conditions_index]:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    if self.n_training_step%100<=5:
                        self.writer_loss(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.base_weight*cond_loss
                    self.list_cond_loss_his_init.append(cond_loss)
                    self.list_cond_loss_his=self.list_cond_loss_his_init
                    list_cond_loss=self.list_cond_loss_his_init
            else:
                list_cond_loss=[]
                for condition in [self.train_conditions[j] for j in train_conditions_index]:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    if self.n_training_step%100<=5:
                        self.writer_loss(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.weight*cond_loss
                    list_cond_loss.append(cond_loss)
            #if self.weight_tunning & ((n_step_init-self.n_training_step)%(self.nsteps*10)==0):
            #    self._baseweight_tunner(list_cond_loss,train_conditions_index)
                if self.weight_tunning & ((n_step_init-self.n_training_step)%self.nsteps==0):
                    self._ReLoBRALO(list_cond_loss,train_conditions_index)
            self.list_cond_loss_his=list_cond_loss
            self.log('train/PINN_loss', loss)
            self.n_training_step += 1
        
            positions=torch.rand(1,self.rand_size)*self.rand_bound
                #####generator
            fake_profiles=self(positions)  #######20 : random profiles
            y_hat=self.discriminator(fake_profiles)
            y=torch.ones(positions.size(1),1)
            g_loss=self.adversarial_loss(y_hat,y)
            self.log('train/G_loss', g_loss)
            if self.n_training_step%100<=5:
                self.writer_loss(f'train/G_loss', g_loss)
            loss=loss+self.GAN_weight*g_loss
            self.log("train/model_loss",loss)
            #if self.n_training_step%1000==0:
                #self._baseweight_tunner()
        if optimizer_idx==1:
            y_hat_real=self.discriminator(real_profiles)
            y_real=torch.ones(real_profiles.size(0),1)
            real_loss=self.adversarial_loss(y_hat_real,y_real)
            y_hat_fake=self.discriminator(self(positions).detach())
            y_fake=torch.zeros(positions.size(1),1)
            fake_loss=self.adversarial_loss(y_hat_fake,y_fake)
            d_loss=(real_loss+fake_loss)/2
            self.log('train/D_loss', d_loss)
            if self.n_training_step%100<=5:
                self.writer_loss(f'train/D_loss', d_loss)
            loss=loss+self.GAN_weight*d_loss
        return loss

    def validation_step(self, batch, batch_idx):
        for condition in self.val_conditions:
            torch.set_grad_enabled(condition.track_gradients is not False)
            self.log(f'val/{condition.name}', condition(device=self.device))

    def configure_optimizers(self):
        optimizer_G = self.optimizer_setting_G.optimizer_class(
            self.generator.parameters(),
            lr = self.optimizer_setting_G.lr,
            **self.optimizer_setting_G.optimizer_args
        )
        optimizer_D = self.optimizer_setting_D.optimizer_class(
            self.discriminator.parameters(),
            lr = self.optimizer_setting_D.lr,
            **self.optimizer_setting_D.optimizer_args
        )
        if self.optimizer_setting_G.scheduler_class is None:
            return [optimizer_G,optimizer_D]

        lr_scheduler_G = self.optimizer_setting_G.scheduler_class(optimizer_G,
            **self.optimizer_setting_G.scheduler_args
        )
        lr_scheduler_G = {'scheduler': lr_scheduler_G, 'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': self.optimizer_setting_G.scheduler_frequency,
                        'monitor': self.optimizer_setting_G.monitor_lr}
        for input_name in self.optimizer_setting_G.scheduler_args:
            lr_scheduler_G[input_name] = self.optimizer_setting_G.scheduler_args[input_name]
            
        lr_scheduler_D = self.optimizer_setting_D.scheduler_class(optimizer_D,
            **self.optimizer_setting_D.scheduler_args
        )
        lr_scheduler_D = {'scheduler': lr_scheduler_D, 'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': self.optimizer_setting_D.scheduler_frequency,
                        'monitor': self.optimizer_setting_D.monitor_lr}
        for input_name in self.optimizer_setting_D.scheduler_args:
            lr_scheduler_D[input_name] = self.optimizer_setting_D.scheduler_args[input_name]
        return [optimizer_G,optimizer_D], [lr_scheduler_G,lr_scheduler_D]


    def forward(self,location):
        #ys=np.linspace(0,2,self.N_points)
        list_x,list_y=torch.meshgrid(location[0],self.ys)
        list_x=list_x.reshape((-1,1))
        list_y=list_y.reshape((-1,1))
        coords = torch.tensor(torch.concat((list_x,list_y),axis=1), dtype=torch.float32)
        output=self.generator(Points(coords, self.co_sys))
        out_matrix=output.as_tensor[:,0:output.dim-1].reshape((len(location[0]),self.N_points*(output.dim-1)))  
        return out_matrix
    
    def adversarial_loss(self,y_hat,y):
        return nn.functional.binary_cross_entropy(y_hat,y)




class Solver_CNN(pl.LightningModule):
    """
    A LightningModule that handles optimization and metric logging of given
    conditions.

    Parameters
    ----------
    train_conditions : tuple or list
        Tuple or list of conditions to be optimized. The weighted sum of their
        losses will be computed and minimized.
    val_conditions : tuple or list
        Conditions to be tracked during the validation part of the training, can
        be used e.g. to track errors comparede to measured data.
    optimizer_setting : OptimizerSetting
        A OptimizerSetting object that contains all necessary parameters for
        optimizing, see :class:`OptimizerSetting`.
    """
    def __init__(self,
                 train_conditions,
                 val_conditions=(),
                 optimizer_setting=OptimizerSetting(torch.optim.Adam,
                                                    1e-3),
                 loss_function_schedule=[{ ##############Modified JY#####################
                        "conditions":[],
                        "max_iter":-1
                    }
                ],
                 weight_tunning=True,
                 weight_tunning_parameters={
                     "alfa":0.99,
                     "E_rho":0.99,
                     "Temperature":1,
                     "tunning_every_n_steps":100
                 }):######################################################################
        super().__init__()
        self.train_conditions = nn.ModuleList(train_conditions)
        self.val_conditions = nn.ModuleList(val_conditions)
        self.optimizer_setting = optimizer_setting
        ############################## Modified JY ######################################
        self.loss_function_schedule=loss_function_schedule
        self.weight_tunning=weight_tunning
        if self.weight_tunning:
            self.alfa=weight_tunning_parameters["alfa"]
            self.E_rho=weight_tunning_parameters["E_rho"]
            self.Temperature=weight_tunning_parameters["Temperature"]
            self.nsteps=weight_tunning_parameters["tunning_every_n_steps"]
        else:
            self.nsteps=0
        ###################################################################################
    def train_dataloader(self):
        """"""
        # HACK: create an empty trivial dataloader, since real data is loaded
        # in conditions
        steps = self.trainer.max_steps
        if steps is None:
            warnings.warn("The maximum amount of iterations should be defined in"
                "trainer.max_steps. If undefined, the solver will train in epochs"
                "of 1000 steps.")
            steps = 1000
        return torch.utils.data.DataLoader(torch.empty(steps))

    def val_dataloader(self):
        """"""
        # HACK: we perform only a single step during validation,
        return torch.utils.data.DataLoader(torch.empty(1))

    def _set_lr_scheduler(self, optimizer):
        lr_scheduler = self.scheduler['class'](optimizer, **self.scheduler['args'])
        lr_scheduler = {'scheduler': lr_scheduler, 'name': 'learning_rate',
                        'interval': 'epoch', 'frequency': 1}
        for input_name in self.scheduler:
            if not input_name in ['class', 'args']:
                lr_scheduler[input_name] = self.scheduler[input_name]
        return lr_scheduler

    def on_train_start(self):
        # move static data to correct device:
        for condition in self.train_conditions:
            condition._move_static_data(self.device)
        for condition in self.val_conditions:
            condition._move_static_data(self.device)
        self.n_training_step = 0

####################Modified JY######################################
###Multi-Objective Loss Balancing for Physics-Informed Deep Learning https://doi.org/10.13140/rg.2.2.20057.24169
    def _ReLoBRALO(self,list_cond_loss,train_conditions_index):   ## RElative LOss Balancing with RAndom LOokback
        m=len(list_cond_loss)
        max_bal=max([list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i]) for i in range(m)])##very large number -- preventing softmax overflow
        sum_exp_L=sum([torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i])-max_bal) for i in range(m)])
        lambda_bal=[self.train_conditions[train_conditions_index[i]].base_weight*m*torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i])-max_bal) / sum_exp_L for i in range(m)]
        max_init=max([list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i]) for i in range(m)])
        sum_exp_L_init=sum([torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i])-max_init) for i in range(m)])
        lambda_bal_init=[self.train_conditions[train_conditions_index[i]].base_weight*m*torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i])-max_init) / sum_exp_L_init for i in range(m)]
        if self.E_rho==1:
            rho=torch.tensor(1)
        else:
            rho=torch.bernoulli(torch.tensor(self.E_rho)) #### all terms share bernoulli random number
        #print(sum_exp_L.item(),sum_exp_L_init.item())
        for i in range(m):
            #print(self.train_conditions[train_conditions_index[i]].name)
            #print(alfa*(rho*self.train_conditions[i].weight+(1-rho)*lambda_bal_init[i]).item(),(1-alfa)*(lambda_bal[i]).item())
            self.train_conditions[train_conditions_index[i]].weight=(self.alfa*(rho*self.train_conditions[train_conditions_index[i]].base_weight+(1-rho)*lambda_bal_init[i])+(1-self.alfa)*lambda_bal[i]).item()
            self.log(f'weight/{self.train_conditions[train_conditions_index[i]].name}', self.train_conditions[train_conditions_index[i]].weight)
        
    def _baseweight_tunner(self,list_cond_loss,train_conditions_index):
        m=len(list_cond_loss)
        for i in range(m):
            factor=torch.bernoulli(torch.tensor(0.2)).item()
            if 1/list_cond_loss[i] >1:
                self.train_conditions[train_conditions_index[i]].base_weight=1/list_cond_loss[i]
            #print(i,self.train_conditions[train_conditions_index[i]].base_weight,factor)
                self.train_conditions[train_conditions_index[i]].base_weight=self.train_conditions[train_conditions_index[i]].base_weight*(10**factor)
            else:
                self.train_conditions[train_conditions_index[i]].base_weight=10**factor
            #print(i,self.train_conditions[train_conditions_index[i]].base_weight)


    def training_step(self, batch, batch_idx): ####################### modified JY ################################
        loss = torch.zeros(1, requires_grad=True, device=self.device)
        ######### first set of loss functions #######
        if self.n_training_step<=self.loss_function_schedule[0]["max_iter"]:   
            train_conditions_index=self.loss_function_schedule[0]["conditions"]
            if self.n_training_step==0:
                self.list_cond_loss_his_init=[]
                self.list_cond_loss_his=[]
                for i in self.train_conditions:
                    i.base_weight=i.weight
                    #i.weight=1
                for condition in [self.train_conditions[j] for j in train_conditions_index]:
                    start=time.time()
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    #print(f"-- {time.time()-start} seconds for condition {condition.name}--")
                    loss = loss + condition.base_weight*cond_loss
                    #self.train_conditions[i].weight=1
                    self.list_cond_loss_his_init.append(cond_loss)
                    self.list_cond_loss_his=self.list_cond_loss_his_init
                    list_cond_loss=self.list_cond_loss_his_init
            else:
                list_cond_loss=[]
                for condition in [self.train_conditions[j] for j in train_conditions_index]:
                    start=time.time()
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    #print(f"-- {time.time()-start} seconds for condition {condition.name}--")
                    loss = loss + condition.weight*cond_loss
                    list_cond_loss.append(cond_loss)
                if self.weight_tunning & (self.n_training_step%self.nsteps==0):
                    self._ReLoBRALO(list_cond_loss,train_conditions_index)
            self.list_cond_loss_his=list_cond_loss
            self.log('train/loss', loss)
            self.n_training_step += 1
            #if self.n_training_step%1000==0:
                #self._baseweight_tunner()
            return loss
        
    
        if self.n_training_step>=self.loss_function_schedule[-1]["max_iter"]:
            train_conditions_index=list(range(len(self.train_conditions)))
            n_step_init=self.loss_function_schedule[-1]["max_iter"]+1
            if self.n_training_step<=(self.loss_function_schedule[-1]["max_iter"]+42):
                self.list_cond_loss_his_init=[]
                self.list_cond_loss_his=[]
                for condition in self.train_conditions:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.base_weight*cond_loss
                    self.list_cond_loss_his_init.append(cond_loss)
                    self.list_cond_loss_his=self.list_cond_loss_his_init
                    list_cond_loss=self.list_cond_loss_his_init
            else:
                list_cond_loss=[]
                for condition in self.train_conditions:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.weight*cond_loss
                    list_cond_loss.append(cond_loss)
                #if self.weight_tunning & ((self.n_training_step-n_step_init)%(self.nsteps*10))==0:
                #    self._baseweight_tunner(list_cond_loss,train_conditions_index)
                if self.weight_tunning &((self.n_training_step-n_step_init)%self.nsteps==0):
                    self._ReLoBRALO(list_cond_loss,train_conditions_index)
            self.list_cond_loss_his=list_cond_loss
            self.log('train/loss', loss)
            self.n_training_step += 1
            
            return loss
        

        for i in range(len(self.loss_function_schedule)-1):
            if (self.n_training_step<=self.loss_function_schedule[i+1]["max_iter"]) & (self.n_training_step>self.loss_function_schedule[i]["max_iter"]):
                train_conditions_index=self.loss_function_schedule[i+1]["conditions"]
                n_step_init=self.loss_function_schedule[i]["max_iter"]+1
                break
        if self.n_training_step < (n_step_init+42):### Running-in buffer
            self.list_cond_loss_his_init=[]
            self.list_cond_loss_his=[]
            for condition in [self.train_conditions[j] for j in train_conditions_index]:
                cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                self.log(f'train/{condition.name}', cond_loss)
                loss = loss + condition.base_weight*cond_loss
                self.list_cond_loss_his_init.append(cond_loss)
                self.list_cond_loss_his=self.list_cond_loss_his_init
                list_cond_loss=self.list_cond_loss_his_init
        else:
            list_cond_loss=[]
            for condition in [self.train_conditions[j] for j in train_conditions_index]:
                cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                self.log(f'train/{condition.name}', cond_loss)
                loss = loss + condition.weight*cond_loss
                list_cond_loss.append(cond_loss)
            #if self.weight_tunning & ((n_step_init-self.n_training_step)%(self.nsteps*10)==0):
            #    self._baseweight_tunner(list_cond_loss,train_conditions_index)
            if self.weight_tunning & ((n_step_init-self.n_training_step)%self.nsteps==0):
                    self._ReLoBRALO(list_cond_loss,train_conditions_index)
        self.list_cond_loss_his=list_cond_loss
        self.log('train/loss', loss)
        self.n_training_step += 1
        
        return loss

    def validation_step(self, batch, batch_idx):
        for condition in self.val_conditions:
            torch.set_grad_enabled(condition.track_gradients is not False)
            self.log(f'val/{condition.name}', condition(device=self.device))

    def configure_optimizers(self):
        optimizer = self.optimizer_setting.optimizer_class(
            self.parameters(),
            lr = self.optimizer_setting.lr,
            **self.optimizer_setting.optimizer_args
        )
        if self.optimizer_setting.scheduler_class is None:
            return optimizer

        lr_scheduler = self.optimizer_setting.scheduler_class(optimizer,
            **self.optimizer_setting.scheduler_args
        )
        lr_scheduler = {'scheduler': lr_scheduler, 'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': self.optimizer_setting.scheduler_frequency,
                        'monitor': self.optimizer_setting.monitor_lr}
        for input_name in self.optimizer_setting.scheduler_args:
            lr_scheduler[input_name] = self.optimizer_setting.scheduler_args[input_name]
        return [optimizer], [lr_scheduler]
    














class PIAN_Solver_CNN(pl.LightningModule):
    """
    A LightningModule that handles optimization and metric logging of given
    conditions.

    Parameters
    ----------
    train_conditions : tuple or list
        Tuple or list of conditions to be optimized. The weighted sum of their
        losses will be computed and minimized.
    val_conditions : tuple or list
        Conditions to be tracked during the validation part of the training, can
        be used e.g. to track errors comparede to measured data.
    optimizer_setting : OptimizerSetting
        A OptimizerSetting object that contains all necessary parameters for
        optimizing, see :class:`OptimizerSetting`.
    """
    def __init__(self,
                 train_conditions,
                                  generator,
                 discriminator,
                 co_sys,
                 disc_space,
                 dist_repository,
                 dist_repository_low,
                 val_conditions=(),
                 optimizer_setting_G=OptimizerSetting(torch.optim.Adam,
                                                    1e-3),
                 optimizer_setting_D=OptimizerSetting(torch.optim.Adam,
                                                    1e-3),
                 loss_function_schedule=[{ ##############Modified JY#####################
                        "conditions":[],
                        "max_iter":-1
                    }
                ],
                 weight_tunning=True,
                 weight_tunning_parameters={
                     "alfa":0.99,
                     "E_rho":0.99,
                     "Temperature":1,
                     "tunning_every_n_steps":100
                 },
                 GAN_weight=1000,
                 N_dist=20,### number of 1d roughness
                 L_x=3.6, ####### lenght of domain
                 N_x=300, ####### n points in x direction for generator
                 N_y=150, ####### n points in y direction for generator
                 Log_dir="runs/"):######################################################################
        super().__init__()
        self.writer_loss=SummaryWriter(Log_dir)
        self.writer_generator=SummaryWriter(Log_dir)
        #######Generator#############
        self.co_sys=co_sys
        self.GAN_weight=GAN_weight
        self.N_dist=N_dist
        self.generator=generator
        self.L_x=L_x
        self.N_x=N_x
        self.N_y=N_y
        self.optimizer_setting_G = optimizer_setting_G
        #############################
        self.discriminator=discriminator
        self.disc_space=disc_space
        self.optimizer_setting_D = optimizer_setting_D
        self.train_conditions = nn.ModuleList(train_conditions)
        self.val_conditions = nn.ModuleList(val_conditions)
        self.repo=dist_repository
        self.repo_low=dist_repository_low
        ############################## Modified JY ######################################
        eta=torch.cos(torch.pi*(torch.tensor([i for i in range(self.N_y)]))/(self.N_y)) 
        ys=(1-eta.detach())
        self.ymesh=ys.reshape((1,1,1,-1)).expand((self.N_dist,1,self.N_x,-1))
        ## produce CNN coordinates
        with torch.no_grad():
            list_x,list_y=torch.meshgrid(torch.linspace(0,self.L_x,self.N_x),ys)
            list_x=list_x.reshape((-1,1))
            list_y=list_y.reshape((-1,1))
            self.coords = torch.tensor(torch.concat((list_x,list_y),axis=1).expand((self.N_dist,self.N_x*self.N_y,2)).reshape((self.N_x*self.N_y*self.N_dist,2)),dtype=torch.float32)
        self.loss_function_schedule=loss_function_schedule
        self.weight_tunning=weight_tunning
        if self.weight_tunning:
            self.alfa=weight_tunning_parameters["alfa"]
            self.E_rho=weight_tunning_parameters["E_rho"]
            self.Temperature=weight_tunning_parameters["Temperature"]
            self.nsteps=weight_tunning_parameters["tunning_every_n_steps"]
        else:
            self.nsteps=0
        ###################################################################################
    def train_dataloader(self):
        """"""
        # HACK: create an empty trivial dataloader, since real data is loaded
        # in conditions
        steps = self.trainer.max_steps
        if steps is None:
            warnings.warn("The maximum amount of iterations should be defined in"
                "trainer.max_steps. If undefined, the solver will train in epochs"
                "of 1000 steps.")
            steps = 1000
        return torch.utils.data.DataLoader(torch.empty(steps))

    def val_dataloader(self):
        """"""
        # HACK: we perform only a single step during validation,
        return torch.utils.data.DataLoader(torch.empty(1))

    def _set_lr_scheduler(self, optimizer):
        lr_scheduler = self.scheduler['class'](optimizer, **self.scheduler['args'])
        lr_scheduler = {'scheduler': lr_scheduler, 'name': 'learning_rate',
                        'interval': 'epoch', 'frequency': 1}
        for input_name in self.scheduler:
            if not input_name in ['class', 'args']:
                lr_scheduler[input_name] = self.scheduler[input_name]
        return lr_scheduler

    def on_train_start(self):
        # move static data to correct device:
        for condition in self.train_conditions:
            condition._move_static_data(self.device)
        for condition in self.val_conditions:
            condition._move_static_data(self.device)
        self.n_training_step = 0

####################Modified JY######################################
###Multi-Objective Loss Balancing for Physics-Informed Deep Learning https://doi.org/10.13140/rg.2.2.20057.24169
    def _ReLoBRALO(self,list_cond_loss,train_conditions_index):   ## RElative LOss Balancing with RAndom LOokback
        m=len(list_cond_loss)
        max_bal=max([list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i]) for i in range(m)])##very large number -- preventing softmax overflow
        sum_exp_L=sum([torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i])-max_bal) for i in range(m)])
        lambda_bal=[self.train_conditions[train_conditions_index[i]].base_weight*m*torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i])-max_bal) / sum_exp_L for i in range(m)]
        max_init=max([list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i]) for i in range(m)])
        sum_exp_L_init=sum([torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i])-max_init) for i in range(m)])
        lambda_bal_init=[self.train_conditions[train_conditions_index[i]].base_weight*m*torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i])-max_init) / sum_exp_L_init for i in range(m)]
        if self.E_rho==1:
            rho=torch.tensor(1)
        else:
            rho=torch.bernoulli(torch.tensor(self.E_rho)) #### all terms share bernoulli random number
        #print(sum_exp_L.item(),sum_exp_L_init.item())
        for i in range(m):
            #print(self.train_conditions[train_conditions_index[i]].name)
            #print(alfa*(rho*self.train_conditions[i].weight+(1-rho)*lambda_bal_init[i]).item(),(1-alfa)*(lambda_bal[i]).item())
            self.train_conditions[train_conditions_index[i]].weight=(self.alfa*(rho*self.train_conditions[train_conditions_index[i]].base_weight+(1-rho)*lambda_bal_init[i])+(1-self.alfa)*lambda_bal[i]).item()
            self.log(f'weight/{self.train_conditions[train_conditions_index[i]].name}', self.train_conditions[train_conditions_index[i]].weight)
        
    def _baseweight_tunner(self,list_cond_loss,train_conditions_index):
        m=len(list_cond_loss)
        for i in range(m):
            factor=torch.bernoulli(torch.tensor(0.2)).item()
            if 1/list_cond_loss[i] >1:
                self.train_conditions[train_conditions_index[i]].base_weight=1/list_cond_loss[i]
            #print(i,self.train_conditions[train_conditions_index[i]].base_weight,factor)
                self.train_conditions[train_conditions_index[i]].base_weight=self.train_conditions[train_conditions_index[i]].base_weight*(10**factor)
            else:
                self.train_conditions[train_conditions_index[i]].base_weight=10**factor
            #print(i,self.train_conditions[train_conditions_index[i]].base_weight)


    def training_step(self, batch, batch_idx,optimizer_idx): ####################### modified JY ################################
        real_profiles,_=batch
        loss = torch.zeros(1, requires_grad=True, device=self.device)
        ######### first set of loss functions #######
        if self.n_training_step<=self.loss_function_schedule[0]["max_iter"]:   
            train_conditions_index=self.loss_function_schedule[0]["conditions"]
            if optimizer_idx==0:
                if self.n_training_step==0:
                    self.list_cond_loss_his_init=[]
                    self.list_cond_loss_his=[]
                    for i in self.train_conditions:
                        i.base_weight=i.weight
                    #i.weight=1
                    for condition in [self.train_conditions[j] for j in train_conditions_index]:
                        cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                        self.log(f'train/{condition.name}', cond_loss)
                        #self.writer_loss(f'train/{condition.name}', cond_loss)
                        loss = loss + condition.base_weight*cond_loss
                        #self.train_conditions[i].weight=1
                        self.list_cond_loss_his_init.append(cond_loss)
                        self.list_cond_loss_his=self.list_cond_loss_his_init
                        list_cond_loss=self.list_cond_loss_his_init
                else:
                    list_cond_loss=[]
                    for condition in [self.train_conditions[j] for j in train_conditions_index]:
                        cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                        self.log(f'train/{condition.name}', cond_loss)
                        loss = loss + condition.weight*cond_loss
                        list_cond_loss.append(cond_loss)
                        #if self.n_training_step%100<=5:
                            #self.writer_loss(f'train/{condition.name}', cond_loss)
                    if self.weight_tunning & (self.n_training_step%self.nsteps==0):
                        self._ReLoBRALO(list_cond_loss,train_conditions_index)
                self.list_cond_loss_his=list_cond_loss
                self.log('train/PINN_loss', loss)
                self.n_training_step += 1
                positions=sample(range(len(self.repo)),self.N_dist)
                #####generator
                fake_profiles=self(self.repo[positions,:],self.repo_low[positions,:])  #######N_dist : random profiles
                y_hat=self.discriminator(fake_profiles)
                y=torch.ones(len(positions),1)
                g_loss=self.adversarial_loss(y_hat,y)
                self.log('train/G_loss', g_loss)
                #if self.n_training_step%100<=5:
                    #self.writer_loss('train/G_loss', g_loss)
                loss=loss+self.GAN_weight*g_loss
                self.log("train/model_loss",loss)
            #if self.n_training_step%1000==0:
                #self._baseweight_tunner()
            elif optimizer_idx==1:
                positions=sample(range(len(self.repo)),self.N_dist)
                y_hat_real=self.discriminator(real_profiles)
                y_real=torch.ones(real_profiles.shape[0],1)
                real_loss=self.adversarial_loss(y_hat_real,y_real)
                y_hat_fake=self.discriminator(self(self.repo[positions,:],self.repo_low[positions,:]).detach())
                y_fake=torch.zeros(len(positions),1)
                fake_loss=self.adversarial_loss(y_hat_fake,y_fake)
                d_loss=(real_loss+fake_loss)/2
                self.log('train/D_loss', d_loss)
                #if self.n_training_step%100<=5:
                    #self.writer_loss('train/D_loss', d_loss)
                loss=loss+self.GAN_weight*d_loss
            return loss
        
    
        if self.n_training_step>=self.loss_function_schedule[-1]["max_iter"]:
            train_conditions_index=list(range(len(self.train_conditions)))
            n_step_init=self.loss_function_schedule[-1]["max_iter"]+1
            if optimizer_idx==0:
                if self.n_training_step<=(self.loss_function_schedule[-1]["max_iter"]+42):
                    self.list_cond_loss_his_init=[]
                    self.list_cond_loss_his=[]
                    for condition in self.train_conditions:
                        cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                        self.log(f'train/{condition.name}', cond_loss)
                        #if self.n_training_step%100<=5:
                            #self.writer_loss(f'train/{condition.name}', cond_loss)
                        loss = loss + condition.base_weight*cond_loss
                        self.list_cond_loss_his_init.append(cond_loss)
                        self.list_cond_loss_his=self.list_cond_loss_his_init
                        list_cond_loss=self.list_cond_loss_his_init
                else:
                    list_cond_loss=[]
                    for condition in self.train_conditions:
                        cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                        self.log(f'train/{condition.name}', cond_loss)
                        #if self.n_training_step%100<=5:
                            #self.writer_loss(f'train/{condition.name}', cond_loss)
                        loss = loss + condition.weight*cond_loss
                        list_cond_loss.append(cond_loss)
                #if self.weight_tunning & ((self.n_training_step-n_step_init)%(self.nsteps*10))==0:
                #    self._baseweight_tunner(list_cond_loss,train_conditions_index)
                    if self.weight_tunning&((self.n_training_step-n_step_init)%self.nsteps==0):
                        self._ReLoBRALO(list_cond_loss,train_conditions_index)
                self.list_cond_loss_his=list_cond_loss
                self.log('train/PINN_loss', loss)
                self.n_training_step += 1
            
                positions=sample(range(len(self.repo)),self.N_dist)
                #####generator
                fake_profiles=self(self.repo[positions,:],self.repo_low[positions,:])  
                y_hat=self.discriminator(fake_profiles)
                y=torch.ones(len(positions),1)
                g_loss=self.adversarial_loss(y_hat,y)
                self.log('train/G_loss', g_loss)
                #if self.n_training_step%100<=5:
                   # self.writer_loss('train/G_loss', g_loss)
                loss=loss+self.GAN_weight*g_loss
                self.log("train/model_loss",loss)
            #if self.n_training_step%1000==0:
                #self._baseweight_tunner()
            if optimizer_idx==1:
                positions=sample(range(len(self.repo)),self.N_dist)
                y_hat_real=self.discriminator(real_profiles)
                y_real=torch.ones(real_profiles.shape[0],1)
                real_loss=self.adversarial_loss(y_hat_real,y_real)
                y_hat_fake=self.discriminator(self(self.repo[positions,:],self.repo_low[positions,:]).detach())
                y_fake=torch.zeros(len(positions),1)
                fake_loss=self.adversarial_loss(y_hat_fake,y_fake)
                d_loss=(real_loss+fake_loss)/2
                self.log('train/D_loss', d_loss)
                #if self.n_training_step%100<=5:
                    #self.writer_loss('train/D_loss', d_loss)
                loss=loss+self.GAN_weight*d_loss
            return loss
        

        for i in range(len(self.loss_function_schedule)-1):
            if (self.n_training_step<=self.loss_function_schedule[i+1]["max_iter"]) & (self.n_training_step>self.loss_function_schedule[i]["max_iter"]):
                train_conditions_index=self.loss_function_schedule[i+1]["conditions"]
                n_step_init=self.loss_function_schedule[i]["max_iter"]+1
                break
        if optimizer_idx==0:
            if self.n_training_step < (n_step_init+42):### Running-in buffer
                self.list_cond_loss_his_init=[]
                self.list_cond_loss_his=[]
                for condition in [self.train_conditions[j] for j in train_conditions_index]:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    #if self.n_training_step%100<=5:
                        #self.writer_loss(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.base_weight*cond_loss
                    self.list_cond_loss_his_init.append(cond_loss)
                    self.list_cond_loss_his=self.list_cond_loss_his_init
                    list_cond_loss=self.list_cond_loss_his_init
            else:
                list_cond_loss=[]
                for condition in [self.train_conditions[j] for j in train_conditions_index]:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    #if self.n_training_step%100<=5:
                        #self.writer_loss(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.weight*cond_loss
                    list_cond_loss.append(cond_loss)
            #if self.weight_tunning & ((n_step_init-self.n_training_step)%(self.nsteps*10)==0):
            #    self._baseweight_tunner(list_cond_loss,train_conditions_index)
                if self.weight_tunning & ((n_step_init-self.n_training_step)%self.nsteps==0):
                    self._ReLoBRALO(list_cond_loss,train_conditions_index)
            self.list_cond_loss_his=list_cond_loss
            self.log('train/PINN_loss', loss)
            self.n_training_step += 1
        
            positions=sample(range(len(self.repo)),self.N_dist)
                #####generator
            fake_profiles=self(self.repo[positions,:],self.repo_low[positions,:])   #######20 : random profiles
            y_hat=self.discriminator(fake_profiles)
            y=torch.ones(len(positions),1)
            g_loss=self.adversarial_loss(y_hat,y)
            self.log('train/G_loss', g_loss)
            #if self.n_training_step%100<=5:
                #self.writer_loss(f'train/G_loss', g_loss)
            loss=loss+self.GAN_weight*g_loss
            self.log("train/model_loss",loss)
            #if self.n_training_step%1000==0:
                #self._baseweight_tunner()
        if optimizer_idx==1:
            positions=sample(range(len(self.repo)),self.N_dist)
            y_hat_real=self.discriminator(real_profiles)
            y_real=torch.ones(real_profiles.shape[0],1)
            real_loss=self.adversarial_loss(y_hat_real,y_real)
            y_hat_fake=self.discriminator(self(self.repo[positions,:],self.repo_low[positions,:]).detach())
            y_fake=torch.zeros(len(positions),1)
            fake_loss=self.adversarial_loss(y_hat_fake,y_fake)
            d_loss=(real_loss+fake_loss)/2
            self.log('train/D_loss', d_loss)
            #if self.n_training_step%100<=5:
                #self.writer_loss(f'train/D_loss', d_loss)
            loss=loss+self.GAN_weight*d_loss
        return loss

    def validation_step(self, batch, batch_idx):
        for condition in self.val_conditions:
            torch.set_grad_enabled(condition.track_gradients is not False)
            self.log(f'val/{condition.name}', condition(device=self.device))

    def configure_optimizers(self):
        optimizer_G = self.optimizer_setting_G.optimizer_class(
            self.generator.parameters(),
            lr = self.optimizer_setting_G.lr,
            **self.optimizer_setting_G.optimizer_args
        )
        optimizer_D = self.optimizer_setting_D.optimizer_class(
            self.discriminator.parameters(),
            lr = self.optimizer_setting_D.lr,
            **self.optimizer_setting_D.optimizer_args
        )
        if self.optimizer_setting_G.scheduler_class is None:
            return [optimizer_G,optimizer_D]

        lr_scheduler_G = self.optimizer_setting_G.scheduler_class(optimizer_G,
            **self.optimizer_setting_G.scheduler_args
        )
        lr_scheduler_G = {'scheduler': lr_scheduler_G, 'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': self.optimizer_setting_G.scheduler_frequency,
                        'monitor': self.optimizer_setting_G.monitor_lr}
        for input_name in self.optimizer_setting_G.scheduler_args:
            lr_scheduler_G[input_name] = self.optimizer_setting_G.scheduler_args[input_name]
            
        lr_scheduler_D = self.optimizer_setting_D.scheduler_class(optimizer_D,
            **self.optimizer_setting_D.scheduler_args
        )
        lr_scheduler_D = {'scheduler': lr_scheduler_D, 'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': self.optimizer_setting_D.scheduler_frequency,
                        'monitor': self.optimizer_setting_D.monitor_lr}
        for input_name in self.optimizer_setting_D.scheduler_args:
            lr_scheduler_D[input_name] = self.optimizer_setting_D.scheduler_args[input_name]
        return [optimizer_G,optimizer_D], [lr_scheduler_G,lr_scheduler_D]

    def construct_mask(self,dist_1d):
        matrix_mask=(dist_1d.reshape((self.N_dist,1,self.N_x,1))> self.ymesh).long()+(dist_1d.reshape((self.N_dist,1,self.N_x,1))> (2- self.ymesh)).long() ### roughness:True void:False
        return matrix_mask
    def forward(self,dist,dist_low):
        #ys=np.linspace(0,2,self.N_points)
        #output=torch.zeros((len(dist)),self.disc_space.dim+1,self.N_x,self.N_y)
        dist_input=dist.expand(self.N_x*self.N_y,self.N_dist,1,dist.shape[2]).transpose(1,0).reshape((-1,1,dist.shape[2]))
        output=torch.permute(self.generator(dist_input,Points(self.coords, self.co_sys)).as_tensor[:,0:5].reshape((self.N_dist,self.N_x,self.N_y,5)),(0,3,1,2))
        #for i in range(len(dist)):
        #    output[i,0:self.disc_space.dim,:,:]=self.generator(dist[i].expand(len(self.N_x*self.N_y),-1),Points(self.coords, self.co_sys)).reshape((self.disc_space.dim,self.N_y,self.N_x).transpose((0,2,1)))
        #output=torch.cat(output,self.construct_mask(dist),1)
        return torch.cat((output,self.construct_mask(dist_low)),1)
    
    def adversarial_loss(self,y_hat,y):
        return nn.functional.binary_cross_entropy(y_hat,y)
    
    

