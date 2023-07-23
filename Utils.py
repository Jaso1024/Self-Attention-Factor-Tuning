import torch.utils.data as data

from PIL import Image
import os
import os.path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import json

_DATASET_NAME = (
    'cifar',
    'caltech101',
    'dtd',
    'oxford_flowers102',
    'oxford_iiit_pet',
    'svhn',
    'sun397',
    'patch_camelyon',
    'eurosat',
    'resisc45',
    'diabetic_retinopathy',
    'clevr_count',
    'clevr_dist',
    'dmlab',
    'kitti',
    'dsprites_loc',
    'dsprites_ori',
    'smallnorb_azi',
    'smallnorb_ele',
)
_CLASSES_NUM = (100, 102, 47, 102, 37, 10, 397, 2, 10, 45, 5, 8, 6, 6, 4, 16, 16, 18, 9)

def get_classes_num(dataset_name):
    dict_ = {name: num for name, num in zip(_DATASET_NAME, _CLASSES_NUM)}
    return dict_[dataset_name]

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)

class CustomDataset(Dataset):
    def __init__(self, data, directory, transform=None):
        self.data = data
        self.transform = transform
        self.directory = directory

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, class_name = self.data[idx]
        image = Image.open(self.directory+img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data(name, evaluate=True, batch_size=64):
    root = './data/' + name
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    with open(root + "/data.json") as f:
        data = json.load(f)

    test_data_loader = DataLoader(
            CustomDataset(data["test"], data['img_path'], transform=transform),
            batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    train_data_loader = DataLoader(
            CustomDataset(data["train"], data['img_path'], transform=transform),
            batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    return train_data_loader, test_data_loader

