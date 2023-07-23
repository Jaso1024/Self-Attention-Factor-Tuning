from SAD import SAD
from Utils import *

if __name__ == "__main__":
    model = SAD(num_classes=get_classes_num('oxford_flowers102'))
    train_dl, test_dl = get_data('oxford_flowers102')
    model.upload_data(train_dl, test_dl)
    model.train(101)