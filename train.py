from model.model import Net
from model.netreader import NetReader
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, nargs=1, help='learning rate')
parser.add_argument('--auto_save', type=int, nargs=1, help='time interval saving model')
parser.add_argument('--num_steps', type=int, nargs=1, help='number of steps')
parser.add_argument('--config', type=str, nargs=1, help='dataset configuration')

args = parser.parse_args()

lr = args.lr
n = args.num_steps
auto_save = args.auto_save
config = json.load(open(args.config))

model = Net()
trainable_var = model.get_trainable()
restore_var = model.get_restorable()
loss = []

for t in range(n):
    '''
    !TODO!
    x - take from reader
    '''
    y_pred = model(x)
    loss.append(loss_fn(y_pred, y))
    '''
    !TODO!
    visualize loss
    '''
    if t % auto_save == 0:
        '''
        save restorable from get_restorable()
        '''
        print(t, loss.data[0])
    
    model.zero_grad()

    loss.backward()
    for param in model.parameters():
        '''
        TODO
        get_trainable()
        '''
        param.data -= learning_rate * param.grad.data


