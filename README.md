# SAFT: Self-Attention Factor-Tuning

A highly efficient fine-tuning technique for large-scale neural networks (16x more parameter-efficient training than regular fine-tuning).

Code for the paper [Self-Attention Factor-Tuning for Parameter-Efficient Fine-Tuning](https://doi.org/10.21203/rs.3.rs-3487308/v2)

[![PyPI Version](https://badge.fury.io/py/saft.svg)](https://badge.fury.io/py/saft)
[![Downloads](https://static.pepy.tech/badge/saft)](https://pepy.tech/project/saft)

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
<div align="center">




![Performance Results](https://github.com/Jaso1024/Self-Attention-Factor-Tuning/assets/107654508/3319cec9-055a-42db-b88b-c2848f3fa907)

![Performance Results](https://github.com/Jaso1024/Self-Attention-Factor-Tuning/assets/107654508/c9f61913-f206-4914-affb-235dec01672b)

![Performance Results](https://github.com/Jaso1024/Self-Attention-Factor-Tuning/assets/107654508/9e53155b-366d-4b54-9b3c-b84fea3561b4)

</div>


