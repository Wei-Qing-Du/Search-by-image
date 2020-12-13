import sys
import os

sys.path.append(os.getcwd() +"\\..\\..\\CNN")
sys.path.append(os.getcwd() + '\\model')
sys.path.append(os.getcwd() + '\\data')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from resnet import ResNet50
from include.data import get_data_set

JSON_CONFIG = 'config.json'
# Load Dataset
train_x, train_y = get_data_set("train")

def train(n_epochs, debug=False):
    # Build model and load data into it
    model = ResNet50(JSON_CONFIG)
    model.build()
    model.train(n_epochs, debug=debug)


def main():
    epochs = 60
    train(epochs)


if __name__ == '__main__':
    main()
