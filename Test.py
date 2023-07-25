from SAD import SAD
from Utils import *
import json
from avalanche.evaluation.metrics.accuracy import Accuracy
import numpy as np
import torch
from torch.optim import AdamW

def check_dataleak(dataset_name='oxford_flowers102'):
    mappings = {
        "caltech101": './data/caltech101/data.json', 
        'oxford_flowers102': './data/oxford_flowers102/data.json', 
        'ucf101':'./data/ucf101/data.json',
        'eurosat': './data/eurosat/data.json',
        'dtd': './data/dtd/data.json'
        }
    train_data = json.load(open(mappings[dataset_name], 'r'))['train']
    test_data = json.load(open(mappings[dataset_name], 'r'))['test']

    # Extract image paths from training and test datasets
    train_image_paths = [item[0] for item in train_data]
    test_image_paths = [item[0] for item in test_data]

    # Check for any common data points between training and test datasets
    common_image_paths = set(train_image_paths) & set(test_image_paths)

    if len(common_image_paths) == 0:
        print(f"No common data points between training and test datasets in {dataset_name}")
    else:
        print(f"Warning: Some data points are present in both training and test datasets in {dataset_name}")
        print("Common data points:", common_image_paths)

def evaluate_model(sad, model, dataset_name='oxford_flowers102'):
    train_dl, test_dl = get_data(dataset_name, evaluate=True)
    sad.upload_data(train_dl, test_dl)
    result = sad.test(model, test_dl)
    return result

def run_model(sad, dataset_name='oxford_flowers102', epochs=101):
    train_dl, val_dl = get_data(dataset_name, evaluate=False)
    sad.upload_data(train_dl, val_dl)
    return sad.train(epochs)


DATASET_NAME = 'caltech101'


if __name__ == "__main__":
        #Make sure no testing data points are in the training data
        #check_dataleak(DATASET_NAME)

        #Non-Timm && Non-ViT models are not supported (yet)
        #Note: SAD uses AdamW with scheduler by default
        sad = SAD(
            model='vit_base_patch16_224_in21k',
            num_classes=get_classes_num(DATASET_NAME),
            validation_interval=10,
            rank=3,
            scale=10,
            timm_ckpt_path='ViT-B_16.npz',
            ckpt_dir='',
            drop_path_rate=.1,
        )

        trained_model = run_model(sad, dataset_name=DATASET_NAME, epochs=101)

        print('Evaluating...')
        print(evaluate_model(sad, sad.model, dataset_name=DATASET_NAME))

