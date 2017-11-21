from model.model import Net
from model.netreader import NetReader
from base.writer import write_summaries
import argparse
import json
import os
import torch.optim as optim
from collections import OrderedDict
from torch.autograd import Variable
import torch

parser = argparse.ArgumentParser()

parser.add_argument('config', type=str, nargs=1, help='dataset configuration')

args = parser.parse_args()

config = json.load(open(args.config[0]))
num_batches = int(config["test_num_batches"])

### model and reader
net_model = Net(config)
net_reader = NetReader(config)
net_model.train(mode=False)

data_loader = net_reader.getDataLoader()
inputs = net_model.get_inputs()
outputs = net_model.get_outputs()
criterion = net_model.get_criterion(config)

### trainable and restorable

restore_var = OrderedDict(net_model.get_restorable())
loss = 0

### optimizer detection


### restoraion 
if config['restore']:
    restore_file = os.path.join(config['restore_path'], config['restore_file'])
    if os.path.isfile(restore_file):
        print("=> loading checkpoint '{}'".format(restore_file))

        checkpoint = torch.load(restore_file)
        start_epoch = checkpoint['epoch']
        net_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})".format(start_epoch, checkpoint['epoch']))
    else:
        print("ERROR:")
        print("=> no checkpoint found at '{}'".format(restore_file))
        exit()

### asserts and assigns
x = dict()
y = dict()

for data in data_loader:

    for key, var in inputs.items():
        if key not in data.keys():
            print("ERROR: In data there is no key - {}".format(key))
            assert(0)
        if data[key].numpy().shape != var.data.numpy().shape:
            print("ERROR: Shapes of inputs and data different at {}".format(key))
            print("shape of data is {}, shape of input is {}".format(data[key].numpy().shape, var.data.numpy().shape))
            assert(0)
        x[key] = Variable(data[key], requires_grad=False)

    for key, var in outputs.items():
        if key not in data.keys():
            print("ERROR: In data there is no key - {}".format(key))
            assert(0)
        y[key] = Variable(data[key], requires_grad=False)
    y_pred = net_model(x)
    
    try:
        loss = criterion(y_pred, y)
    except:
        print("ERROR:") 
        print("loss name is {}".format(str(loss)))
        for key, var in outputs.items():
            print("key: {}".format(key))
            print("shape of data is {}".format(data[key].numpy().shape))
            print("type of data is {}".format(data[key].type()))
            print("shape of input is {}".format(var.data.numpy().shape))
            print("type of input is {}".format(var.data.type()))
            exit()
    break

### testing

t = 0
for data in data_loader:
    if t >= num_batches:
        exit()

    for key, var in inputs.items():
        x[key].data = data[key]

    for key, var in outputs.items():
        y[key].data = data[key]

    y_pred = net_model(x)
    
    loss = criterion(y_pred, y)
    loss_float = loss.data.numpy()[0]
    
    net_reader.visualize(x, y_pred, loss_float)

    print("-----------------------------------------------\n")
    print("step: {}\n losses:\n {}\n".format(t, loss.data[0]))
    t += 1

