# SAD
Implementation of Self-Attention Tensor-Decomposition, an original technique for Parameter-Efficient Fine-Tuning 

## Requirements
```
- Python == 3.8
- torch == 1.10.0
- torchvision == 0.11.1
- timm == 0.4.12
- avalanche-lib == 0.1.0
```

## Data
To get started, head over to the SSF data preparation page and download the VTAB-1K dataset: [SSF Data Preparation](https://github.com/dongzelian/SSF#data-preparation). Once you've obtained the data, place the dataset folders into `<YOUR PATH>/SAD/data/`.

## Pretrained Model
Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to `<YOUR PATH>/SAD/ViT-B_16.npz`


## Results (.055 Million Trainable Backbone Parameters) (ViT-B/16)
Coming soon...
```
Caltech101: 90.6%
```

## Run
```sh
cd <YOUR PATH>/SAD
python3 Test.py
```

## Modular Quick-Start
```python
from SAD import SAD
from Utils import *

if __name__ == "__main__":
    sad = SAD(
        model='vit_base_patch16_224',
        num_classes=get_classes_num('oxford_flowers102'),
        validation_interval=1,
        rank=3,
        scale=10
    )
    train_dl, test_dl = get_data('oxford_flowers102')
    sad.upload_data(train_dl, test_dl)
    sad.train(10)
```

