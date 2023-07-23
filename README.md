# SAD
Implementation of Self-Attention Tensor-Decomposition

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
- Add property "img_path" to the new data.json file, with value "data/<dataset name>/<folder containing raw data>/"


## Quick-Start
```
from SAD import SAD
from Utils import *

if __name__ == "__main__":
    model = SAD(cuda=False, num_classes=get_classes_num('caltech101'))
    train_dl, test_dl = get_data('caltech101')
    model.upload_data(train_dl, test_dl)
    model.train(10)
```

