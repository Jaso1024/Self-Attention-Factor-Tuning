from SAD import SAD
from Utils import *

if __name__ == "__main__":
    model = SAD(cuda=False, num_classes=get_classes_num('caltech101'))
    train_dl, test_dl = get_data('caltech101')
    model.upload_data(train_dl, test_dl)
    model.train(10)