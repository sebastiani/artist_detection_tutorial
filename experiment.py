import os
import pickle
import yaml
from argparse import ArgumentParser
from src.BaseModel import BaseModel
from src.FullyConnectedArch import FullyConnectedModel
from src.CNNArch import CNNModel
#from torchsummary import summary

parser = ArgumentParser()
parser.add_argument('--mode', type=str, help='train/infer')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--data', type=str, help='data to infer on, etiher this or define on a config file')
args = parser.parse_args()

def main():
    # loading config file
    with open(args.config, 'r') as f:
        params = yaml.load(f)

    trainConfig = params['training']
    evalConfig = params['evaluation']
    model = BaseModel(trainConfig)

    if trainConfig['model'] == 'CNNArch':
        model.model = CNNModel(trainConfig)

    elif trainConfig['model'] == 'FullyConnectedModel':
        model.model = FullyConnectedModel(trainConfig)

    else:
        raise("Model specified not found")
    #summary(model.model.cuda(), (3, 256, 256))
    if args.mode == 'train':
        model.train(trainConfig)

    elif args.mode == 'infer':
        try:
            data = evalConfig['data']
        except KeyError:
            if args.data is None:
                raise('No data specified to infer on')
            else:
                with open(args.data, 'rb') as f:
                    data = pickle.load(f)

        model.predict(evalConfig)

if __name__ == '__main__':
    main()
