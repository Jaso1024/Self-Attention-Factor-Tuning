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
To get started, head over to the data preparation page: [NOAH Data Preparation]([https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)). Once you've obtained the data, place the dataset folders into `<YOUR PATH>/SAD/data/`.

For each dataset, make sure to:
- Rename split_zhou_<Dataset Name>.json to data.json
- Add property "img_path" to the new data.json file, with value `data/<DATASET NAME>/<DATA-CONTAINING FOLDER NAME>/`

## Results (.055 Million Trainable Backbone Parameters) (ViT-B/16)
Coming soon...
```
```

## Quick-Start
```
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

