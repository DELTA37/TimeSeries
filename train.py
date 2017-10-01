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

parser.add_argument('lr', type=float, nargs=1, help='learning rate')
parser.add_argument('auto_save', type=int, nargs=1, help='time interval saving model')
parser.add_argument('num_steps', type=int, nargs=1, help='number of steps')
parser.add_argument('num_epoch', type=int, nargs=1, help='number of epochs')
parser.add_argument('opt', type=str, nargs=1, help='optimizer')
parser.add_argument('config', type=str, nargs=1, help='dataset configuration')

args = parser.parse_args()

def save_checkpoint(state, path, epoch=0, step=0):
    filename = os.path.join(path, 'checkpoint-{}-{}.ckpt.tar'.format(epoch, step))
    torch.save(state, filename)
   
lr = args.lr[0]
n = args.num_steps[0]
N = args.num_epoch[0]
auto_save = args.auto_save[0]
config = json.load(open(args.config[0]))
start_epoch = 0
opt_name = args.opt[0]

### model and reader
net_model = Net()
net_reader = NetReader(config)

data_loader = net_reader.getDataLoader()
inputs = net_model.get_inputs()
outputs = net_model.get_outputs()
criterion = net_model.get_criterion(config)

### trainable and restorable
trainable_var = OrderedDict(net_model.get_trainable())
untrainable_var = OrderedDict(net_model.named_parameters())

for key, val in trainable_var.items():
    del untrainable_var[key]

restore_var = OrderedDict(net_model.get_restorable())
loss = 0

### optimizer detection

opt_params = [
    {
        'params': list(trainable_var.values()),
        'lr' : lr,
    },
    {
        'params': list(untrainable_var.values()),
        'lr' : 0,
    },
]

if opt_name == 'SGD':
    optimizer = optim.SGD(opt_params)
    closure_bool = 0
elif opt_name == 'Adam':
    optimizer = optim.Adam(opt_params)
    closure_bool = 0
else:
    print("ERROR:")
    print("There is no optimizer named {}".format(opt_name))

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
else:
    if not os.path.isdir(config['restore_path']):
        os.mkdir(config['restore_path'])

### training

for ep in range(start_epoch, start_epoch + N):
    t = 0
    for data in data_loader:
        x = dict()
        ''' # for a dict values passed to __call__
        y = dict()
        '''
        for key, var in inputs.items():
            if key not in data.keys():
                print("ERROR: In data there is no key - {}".format(key))
                assert(0)
            if data[key].numpy().shape != var.data.numpy().shape:
                print("ERROR: Shapes of inputs and data different at {}".format(key))
                print("shape of data is {}, shape of input is {}".format(data[key].numpy().shape, var.data.numpy().shape))
                assert(0)
            x[key] = Variable(data[key], requires_grad=False)
        ''' # for a dict values passed to __call__
        for key, var in outputs.items():
            if key not in data.keys():
                print("ERROR: In data there is no key - {}".format(key))
                assert(0)
            if data[key].numpy().shape != var.data.numpy().shape:
                print("ERROR: Shapes of inputs and data different at {}".format(key))
                print("shape of data is {}, shape of input is {}".format(data[key].numpy().shape, var.data.numpy().shape))
                assert(0)
            y[key] = Variable(data[key], requires_grad=False)
        '''
        y = Variable(data['label'], requires_grad=False)

        def closure(): # special opt methods
            if not hasattr(closure, 'once'): 
                '''
                we need once as we send this function to optimiser 
                and it can invoke function several times
                '''
                closure.once = 1

            optimizer.zero_grad()

            y_pred = net_model(x)
            
            loss = criterion(y_pred, y)

            loss_float = loss.data.numpy()[0]

            if closure.once and t % auto_save == 0:

                write_summaries({'loss':loss_float}, config['summary_path'])

                d = {
                    'epoch' : ep + 1,
                    'state_dict' : net_model.get_restorable(),
                    'optimizer' : optimizer.state_dict()
                }
                save_checkpoint(d, config['restore_path'], ep, t)
                print("-----------------------------------------------\n")
                print("step: {}\n losses:\n {}\n".format(t, loss.data[0]))

            loss.backward()
            closure.once = 0
            return loss

        if closure_bool:
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()
        closure.once = 1
        t += 1
        

