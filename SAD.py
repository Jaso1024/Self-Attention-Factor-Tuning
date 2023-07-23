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

class SAD():
    def __init__(
        self, 
        model=None, 
        optimizer=None, 
        num_classes=12, 
        rank=2, 
        scale=1, 
        match_dim=192, 
        learning_rate=1e-3,  
        verbose=True,
        cuda=True,
        seed=42,
        ckpt_dir='',
        weight_decay=1e-4, 
        cycle_decay=.9,
        t_initial=100, 
        warmup_t=10, 
        lr_min=1e-5, 
        warmup_lr_init=1e-6,
        ):

        self.set_seed(seed)

        if model is None:
            self.model = timm.create_model('vit_tiny_patch16_224')
            self.model.reset_classifier(num_classes)
        else:
            self.model = model
        

        self.rank = rank
        self.scale = scale
        self.match_dim = match_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.trainable_params = []
        self.num_trainable_params = 0
        self.num_total_params = 0
        self.verbose = verbose
        self.cuda = cuda
        self.train_loader = None
        self.test_loader = None
        self.best_accuracy = 0
        self.ckpt_dir = ckpt_dir

        self.decompose_attention(self.model)
        self.set_trainable_params()



        if optimizer is None:
            self.optimizer = AdamW(
                self.trainable_params, 
                lr=self.learning_rate,
                 weight_decay=self.weight_decay
            )

            self.scheduler = CosineLRScheduler(
                self.optimizer, 
                t_initial=t_initial, 
                warmup_t=warmup_t, 
                lr_min=lr_min, 
                warmup_lr_init=warmup_lr_init, 
                cycle_decay=cycle_decay
            )
        
        if self.verbose:
            print(f'Number of Trainable Parameters: {self.num_trainable_params} | Number of Total Parameters: {self.num_total_params} | % Trainable Parameters: {self.num_trainable_params/self.num_total_params}')


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
                self.num_trainable_params += parameter.numel()
            else:
                parameter.requires_grad = False
            self.num_total_params += parameter.numel()
            

    def decomposed_forward(self, model, x):
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
        projection = model.projection_SAD(x)
        projection += model.v_SAD(model.drop(model.projection_SAD(model.u_SAD(x)))) * model.scale
        x = model.proj_drop(projection)
        return x
    
    def check_for_data(self):
        if self.train_loader is None or self.test_loader is None:
            assert True == False, "Please make sure to upload your data to the module using the upload_data() function before training/testing"

    def upload_data(self, train, test=None):
        assert type(train) == DataLoader, "Please make sure that the training Dataset is of type Torch.utils.data.DataLoader"
        self.train_loader = train
        
        assert type(test) == DataLoader, "Please make sure that the training Dataset is of type Torch.utils.data.DataLoader"
        self.test_loader = test

    def train(self, epochs):
        self.check_for_data()
        if self.cuda:
            model = model.cuda()
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
            
            if epoch % 2 == 0:
                acc = self.test(self.model, self.test_loader)
                if acc > self.best_accuracy:
                    self.best_accuracy = acc
                    self.save(model, acc, epoch)
                    pbar.set_description(f'Running Loss: {sum(loss_list)/len(loss_list)} | Loss: {loss.item()} | Best Accuracy: {self.best_accuracy} | Accuracy: {str(acc)}')

        self.model = model.cpu()
        return self.model

    def test(self):
        with torch.no_grad():
            self.model.eval()
            acc = Accuracy()
            if self.cuda:
                self.model = self.model.cuda()
            for batch in self.test_set:  # pbar:
                if self.cuda:
                    x, y = batch[0].cuda(), batch[1].cuda()
                else:
                    x, y = batch[0], batch[1]
                out = self.model(x).data
                acc.update(out.argmax(dim=1).view(-1), y)

        return acc.result()
    
    def save(self, model, acc, ep):
        with torch.no_grad():
            model.eval()
            model = model.cpu()
            trainable = {}
            for name, parameter in model.named_parameters():
                if 'SAD' in name or 'head' in name:
                    trainable[name] = parameter.data
            torch.save(trainable, 'checkpoints' +  + '.pt')
            with open(self.ckpt_dir  + '.log', 'w') as file:
                file.write(str(ep) + ' ' + str(acc))

    def decompose_attention(model, self):
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
                attention.scale = self.scale
                attention.dim = self.rank
                bound_method = self.decomposed_forward.__get__(attention, attention.__class__)
                setattr(attention, 'forward', bound_method)

            for child in self.model.children():
                self.decompose_attention(child)
            
