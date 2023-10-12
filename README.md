# SAFT: Self-Attention Factor-Tuning

A highly efficient fine-tuning technique for large-scale neural networks.

[![PyPI Version](https://badge.fury.io/py/saft.svg)](https://badge.fury.io/py/saft)

## Table of Contents

- [Quickstart](#quickstart)
- [VTAB-1k Test](#vtab-1k-test)
- [Pretrained Model](#pretrained-model)
- [Results](#results)

## Quickstart

Easily install SAFT using pip and get started with a simple example.

### Installation

```sh
pip install saft
```

### Example Usage

```python
from saft.saft import saft

if __name__ == "__main__":
    saft_instance = saft(
        model='vit_base_patch16_224',
        num_classes=get_classes_num('oxford_flowers102'),
        validation_interval=1,
        rank=3,
        scale=10
    )
    # Replace with your PyTorch DataLoader objects
    # train_dl, test_dl = [your data in a pytorch dataloader]
    # saft_instance.upload_data(train_dl, test_dl)
    
    saft_instance.train(10)
    trained_model = saft_instance.model
```

## VTAB-1k Test

To run tests on the VTAB-1K dataset, follow these steps:

1. Visit the [SSF Data Preparation](https://github.com/dongzelian/SSF#data-preparation) page to download the VTAB-1K dataset.
2. Place the downloaded dataset folders in `<YOUR PATH>/SAFT/data/`.

## Pretrained Model

For a quick start, download the pretrained ViT-B/16 model:

- [Download ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz)
- Place the downloaded model in `<YOUR PATH>/SAFT/ViT-B_16.npz`.

## Results

Achieve remarkable performance with only 0.055 million trainable backbone parameters using ViT-B/16.

![Performance Results](https://github.com/Jaso1024/SAFT/assets/107654508/2ca64ade-2442-4767-9736-5a3c39ef04cc)
