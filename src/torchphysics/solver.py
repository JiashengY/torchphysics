from typing import Dict
import warnings

import torch
import torch.nn as nn
import pytorch_lightning as pl


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
                 loss_function_schedule=[{
                        "conditions":[],
                        "max_iter":-1
                    }
                ],
                 weight_tunning=True,
                 weight_tunning_parameters={
                     "alfa":0.99,
                     "E_rho":0.99,
                     "Temperature":1
                 }):
        super().__init__()
        self.train_conditions = nn.ModuleList(train_conditions)
        self.val_conditions = nn.ModuleList(val_conditions)
        self.optimizer_setting = optimizer_setting
        self.loss_function_schedule=loss_function_schedule
        self.weight_tunning=weight_tunning
        if self.weight_tunning:
            self.alfa=weight_tunning_parameters["alfa"]
            self.E_rho=weight_tunning_parameters["E_rho"]
            self.Temperature=weight_tunning_parameters["Temperature"]
            self.nsteps=weight_tunning_parameters["tunning_every_n_steps"]
        else:
            self.nsteps=0
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


###Multi-Objective Loss Balancing for Physics-Informed Deep Learning https://doi.org/10.13140/rg.2.2.20057.24169
    def _ReLoBRALO(self,list_cond_loss,train_conditions_index):   ## RElative LOss Balancing with RAndom LOokback
        m=len(list_cond_loss)
        max_bal=max([list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i]) for i in range(m)])##very large number -- preventing softmax overflow
        sum_exp_L=sum([torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i])-max_bal) for i in range(m)])
        lambda_bal=[self.train_conditions.base_weight*m*torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his[i])-max_bal) / sum_exp_L for i in range(m)]
        max_init=max([list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i]) for i in range(m)])
        sum_exp_L_init=sum([torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i])-max_init) for i in range(m)])
        lambda_bal_init=[self.train_conditions.base_weight*m*torch.exp(list_cond_loss[i]/(self.Temperature*self.list_cond_loss_his_init[i])-max_init) / sum_exp_L_init for i in range(m)]
        rho=torch.bernoulli(torch.tensor(self.E_rho)) #### all terms share bernoulli random number
        #print(sum_exp_L.item(),sum_exp_L_init.item())
        for i in range(m):
            #print(self.train_conditions[train_conditions_index[i]].name)
            #print(alfa*(rho*self.train_conditions[i].weight+(1-rho)*lambda_bal_init[i]).item(),(1-alfa)*(lambda_bal[i]).item())
            self.train_conditions[train_conditions_index[i]].weight=(self.alfa*(rho*self.train_conditions[i].weight+(1-rho)*lambda_bal_init[i])+(1-self.alfa)*lambda_bal[i]).item()
            self.log(f'weight/{self.train_conditions[train_conditions_index[i]].name}', self.train_conditions[train_conditions_index[i]].weight)
        
        


    def training_step(self, batch, batch_idx):
        loss = torch.zeros(1, requires_grad=True, device=self.device)
        ######### first set of loss functions #######
        if self.n_training_step<=self.loss_function_schedule[0]["max_iter"]:   
            train_conditions_index=self.loss_function_schedule[0]["conditions"]
            if self.n_training_step==0:
                self.list_cond_loss_his_init=[]
                self.list_cond_loss_his=[]
                for i in self.train_conditions:
                    i.base_weight=i.weight
                    i.weight=1
                for condition in [self.train_conditions[j] for j in train_conditions_index]:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.weight*cond_loss
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
            return loss
        
    
        if self.n_training_step>=self.loss_function_schedule[-1]["max_iter"]:
            train_conditions_index=self.loss_function_schedule[-1]["conditions"]
            n_step_init=self.loss_function_schedule[-1]["max_iter"]+1
            if self.n_training_step<=(self.loss_function_schedule[-1]["max_iter"]+42):
                self.list_cond_loss_his_init=[]
                self.list_cond_loss_his=[]
                for condition in self.train_conditions:
                    cond_loss =  condition(device=self.device, iteration=self.n_training_step)
                    self.log(f'train/{condition.name}', cond_loss)
                    loss = loss + condition.weight*cond_loss
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
                if self.weight_tunning&((n_step_init-self.n_training_step)%self.nsteps==0):
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
                loss = loss + condition.weight*cond_loss
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
