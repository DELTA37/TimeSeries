import argparse
import torch
import torch.nn as nn
from contrib.shape_utils import ShapeOutput
from model.model import Net
from model.netreader import NetReader
import json
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument("--shape", nargs=1, help="In shape", type=str)
parser.add_argument("--reader", nargs=1, help="Reader shape", type=bool)
parser.add_argument("--layer", nargs=1, help="Layer shape", type=str)
parser.add_argument("--model", nargs=1, help="Model shape", type=bool)
parser.add_argument("--config", type=str, nargs=1, help='dataset configuration')

args = parser.parse_args()

if args.model:
    config = json.load(open(args.config[0]))
    net_model = Net(config)
    inputs = net_model.get_inputs()
    outputs = net_model.get_outputs()
    for key, var in inputs.items():
        print("shape of {} is {}".format(key, var.data.numpy().shape))
    
    for key, var in outputs.items():
        print("shape of {} is {}".format(key, var.data.numpy().shape))

elif args.reader:
    config = json.load(open(args.config[0]))
    net_reader = NetReader(config)
    obj0 = net_reader.dataset[0]
    for key, var in obj0.items():
        if isinstance(var, int):
            print("shape of {} is {}".format(key, (0,)))
        if isinstance(var, np.ndarray):
            print("shape of {} is {}".format(key, var.shape))
        if isinstance(var, torch.Tensor):
            print("shape of {} is {}".format(key, tuple(var.size())))
        if isinstance(var, torch.autograd.Variable):
            print("shape of {} is {}".format(key, var.data.numpy().shape))
            
elif args.layer:
    shape = eval(''.join(args.shape[0]))
    layer_name = args.layer[0]
    print("In shape: {}\nOut shape: {}".format(shape, ShapeOutput(tuple(shape), layer_name)))

