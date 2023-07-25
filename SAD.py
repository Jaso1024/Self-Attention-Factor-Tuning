import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torch.optim import AdamW
from torchvision import transforms
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
import numpy as np
import random
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
import yaml
from Utils import *

def decomposed_forward(model, x):
        B, N, C = x.shape
        qkv = model.qkv(x)

        query = model.v_SAD(model.drop(model.query_SAD(model.u_SAD(x))))
        key = model.v_SAD(model.drop(model.key_SAD(model.u_SAD(x))))
        value = model.v_SAD(model.drop(model.value_SAD(model.u_SAD(x))))

        qkv += torch.cat([query, key, value], dim=2) * model.s

        qkv = qkv.reshape(B, N, 3, model.num_heads, C // model.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = (query @ key.transpose(-2, -1)) * model.scale
        attention = attention.softmax(dim=-1)
        attention = model.attn_drop(attention)

        x = (attention @ value).transpose(1, 2).reshape(B, N, C)
        projection = model.proj(x)
        projection += model.v_SAD(model.drop(model.projection_SAD(model.u_SAD(x)))) * model.scale
        x = model.proj_drop(projection)
        return x

class SAD():
    def __init__(
        self, 
        model=None, 
        optimizer=None, 
        load=False,
        num_classes=12, 
        rank=3, 
        scale=10,  
        learning_rate=1e-3,  
        verbose=True,
        validation_interval=1,
        validation_start=10,
        cuda=True,
        save=True,
        seed=42,
        ckpt_dir='',
        weight_decay=None, 
        scheduler = CosineLRScheduler,
        cycle_decay=.9,
        t_initial=100, 
        warmup_t=10, 
        lr_min=1e-5, 
        warmup_lr_init=1e-6,
        timm_ckpt_path=None,
        drop_path_rate=.1,
        best_acc=0,
        train_loader=None,
        test_loader=None
        ):

        self.set_seed(seed)

        if model is None:
            if type(timm_ckpt_path) == str:
                self.model = timm.create_model('vit_base_patch16_224', checkpoint_path=timm_ckpt_path)
            else:
                self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.model.reset_classifier(num_classes)
        elif type(model) is str:
            self.model = timm.create_model(model, pretrained=True, drop_path_rate=drop_path_rate)
            self.model.reset_classifier(num_classes)
        else:
            self.model = model
        
        self.rank = rank
        self.model_name = str(model)
        self.scale = scale
        self.match_dim = self.model.blocks[0].norm1.normalized_shape[0]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay if weight_decay is not None else 1e-4
        self.trainable_params = []
        self.num_trainable_params = 0
        self.num_total_params = 0
        self.verbose = verbose
        self.cuda = cuda
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_accuracy = best_acc
        self.ckpt_dir = ckpt_dir
        self.validation_interval = validation_interval
        self.validation_start = validation_start
        self.save = save

        self.model = self.decompose_attention(self.model)
        self.set_trainable_params()

        if optimizer is None:
            if weight_decay is None:
                self.optimizer = AdamW(
                    self.trainable_params, 
                    lr=self.learning_rate,
                    weight_decay=1e-4
                )
            else:
                self.optimizer = AdamW(
                    self.trainable_params, 
                    lr=self.learning_rate,
                    weight_decay=weight_decay
                )
        else:
            if weight_decay is None:
                self.optimizer = optimizer(
                    self.trainable_params, 
                    lr=self.learning_rate,
                )
            else:
                self.optimizer = optimizer(
                    self.trainable_params, 
                    lr=self.learning_rate,
                    weight_decay=weight_decay
                )

        if scheduler is not None:
            self.scheduler = CosineLRScheduler(
                self.optimizer,
                cycle_decay=cycle_decay,
                t_initial=t_initial, 
                warmup_t=warmup_t, 
                lr_min=lr_min, 
                warmup_lr_init=warmup_lr_init,
            )
        else:
            self.scheduler = None

        if self.verbose:
            print(f'Number of Trainable Parameters (in backbone): {self.num_trainable_params} | Number of Total Parameters: {self.num_total_params} | % Trainable Parameters: {self.num_trainable_params/self.num_total_params}')

        if load:
            self.load_model(self.ckpt_dir)

    def set_seed(self, seed=0):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set_trainable_params(self):
        for name, parameter in self.model.named_parameters():
            if 'SAD' in name or 'head' in name:
                self.trainable_params.append(parameter)
                if 'SAD' in name:
                    self.num_trainable_params += parameter.numel()
            else:
                parameter.requires_grad = False
            self.num_total_params += parameter.numel()
        
        
            

    def save_model(self, model_path):
        if self.model is None:
            raise ValueError("Model is not initialized. Please train the model first.")
        torch.save(self.model.state_dict(), model_path + f'{self.model_name}.pt')

    def load_model(self, model_path):
        if self.model is None:
            raise ValueError("Model is not initialized. Please create or load the model before calling load_model().")
        self.model.load_state_dict(torch.load(model_path + f'{self.model_name}.pt'))
        print(f"Model loaded from {model_path}")
        return self.model
    
    def check_for_data_train(self):
        if self.train_loader is None:
            assert True == False, "Please make sure to upload your data to the module using the upload_data() function before training"

    def check_for_data_train(self):
        if self.test_loader is None:
            assert True == False, "Please make sure to upload your data to the module using the upload_data() function before testing"
        

    def upload_data(self, train, test=None):
        assert type(train) == DataLoader, "Please make sure that the training Dataset is of type Torch.utils.data.DataLoader"
        self.train_loader = train
        if test is not None:
            assert type(test) == DataLoader, "Please make sure that the training Dataset is of type Torch.utils.data.DataLoader"
            self.test_loader = test

    def train(self, epochs):
        self.check_for_data_train()
        if self.cuda:
            self.model = self.model.cuda()
        pbar = tqdm(range(epochs))
        acc = 0
        for epoch in pbar:
            self.model.train()
            if self.cuda:
                self.model = self.model.cuda()
            loss_list = []
            for index, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                if self.cuda:
                    x, y = batch[0].cuda(), batch[1].cuda()
                else:
                    x, y = batch[0], batch[1]
                
                out = self.model(x)
                loss = F.cross_entropy(out, y)

                
                loss.backward()
                self.optimizer.step()
                
                loss_list.append(loss.item())
                pbar.set_description(f'Running Loss: {sum(loss_list)/len(loss_list)} | Loss: {loss.item()} | Best Accuracy: {self.best_accuracy} | Accuracy: {str(acc)}')

            if self.scheduler is not None:
                self.scheduler.step(epoch)
            
            if epoch % self.validation_interval == 0 and epoch > self.validation_start-1:
                acc = self.test(self.model, self.test_loader)
                if acc > self.best_accuracy:
                    self.best_accuracy = acc
                    if self.save:
                        self.save_model(self.ckpt_dir)
                    pbar.set_description(f'Running Loss: {sum(loss_list)/len(loss_list)} | Loss: {loss.item()} | Best Accuracy: {self.best_accuracy} | Accuracy: {str(acc)}')

        self.model = self.model.cpu()
        return self.model

    
    def test(self, model, loader):
        with torch.no_grad():
            model.eval()
            acc = Accuracy()
            if self.cuda:
                model = model.cuda()
            for batch in loader: 
                if self.cuda:
                    x, y = batch[0].cuda(), batch[1].cuda()
                else:
                    x, y = batch[0], batch[1]
                out = model(x).data
                acc.update(out.argmax(dim=1).view(-1), y)

        return acc.result()
    
    

    def decompose_attention(self, model):
        if type(model) == timm.models.vision_transformer.VisionTransformer:
            for block in self.model.blocks:
                attention = block.attn

                attention.u_SAD = nn.Linear(self.match_dim, self.rank, bias=False)
                attention.v_SAD = nn.Linear(self.rank, self.match_dim, bias=False)
                nn.init.zeros_(attention.v_SAD.weight)

                attention.query_SAD = nn.Linear(self.rank, self.rank, bias=False)
                attention.key_SAD = nn.Linear(self.rank, self.rank, bias=False)
                attention.value_SAD = nn.Linear(self.rank, self.rank, bias=False)
                attention.projection_SAD = nn.Linear(self.rank, self.rank, bias=False)
                attention.drop = nn.Dropout(0.1)
                attention.s = self.scale
                attention.dim = self.rank
                bound_method = decomposed_forward.__get__(attention, attention.__class__)
                setattr(attention, 'forward', bound_method)

            for child in model.children():
                child = self.decompose_attention(child)
        
        return model
            