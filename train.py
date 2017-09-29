from model.model import Net
from model.netreader import NetReader
from base.writer import write_summaries
import argparse
import json
import torch.optim as optim

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
model = Net()
reader = NetReader(config)
criterion = model.get_criterion()

### trainable and restorable
trainable_var = model.get_trainable()
restore_var = model.get_restorable()
loss = 0

### optimizer detection

if opt_name == 'SGD':
    optimizer = optim.SGD(trainable_var, lr=lr)
    closure_bool = 0
elif opt_name == 'Adam':
    optimizer = optim.Adam(trainable_var, lr=lr)
    closure_bool = 0
else:
    print("ERROR:")
    print("There is no optimizer named {}".format(opt_name))

### restoraion

if os.path.isfile(config['restore_path']):
    print("=> loading checkpoint '{}'".format(config['restore_path']))

    checkpoint = torch.load(config['restore_path'])
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> loaded checkpoint '{}' (epoch {})".format(start_epoch, checkpoint['epoch']))
else:
    print("ERROR:")
    print("=> no checkpoint found at '{}'".format(config['restore_path']))
    exit()

### training

for ep in range(start_epoch, start_epoch + N):
    for t in range(n):

        data_x, data_y = reader.get_batch(config['batch_size'])
        x, y = Variable(data_x), Variable(data_y)
        def closure(): # special opt methods
            if not hasattr(closure, 'once'): 
                '''
                we need once as we send this function to optimiser 
                and it can invoke function several times
                '''
                closure.once = 1

            optimizer.zero_grad()
            y_pred = model(x)
            
            loss = criterion(y_pred, y)

            if closure.once and t % auto_save == 0:

                write_summaries({'loss':loss}, config['summary_path'])

                d = {
                        'epoch' : ep + 1,
                        'state_dict' : model.get_restorable(),
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
            optimizer.step()
        closure.once = 1
        

