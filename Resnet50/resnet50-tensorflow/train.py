import argparse
import sys
import os

#sys.path.append('../../CNN/include/')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model.resnet import ResNet50
from data import get_data_set

JSON_CONFIG = 'config.json'
# Load Dataset
train_x, train_y = get_data_set("train")

def train(n_epochs, debug=False):
    # Build model and load data into it
    model = ResNet50(JSON_CONFIG, n_classes)
    model.build()
    model.train(n_epochs, debug=debug)


def main():
    epochs = 60
    train(folder, epochs, debug)


if __name__ == '__main__':
    main()
