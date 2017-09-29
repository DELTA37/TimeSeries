from model.model import Net
from model.netreader import NetReader
from base.writer import write_summaries
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, nargs=1, help='learning rate')
parser.add_argument('--auto_save', type=int, nargs=1, help='time interval saving model')
parser.add_argument('--num_steps', type=int, nargs=1, help='number of steps')
parser.add_argument('--num_epoch', type=int, nargs=1, help='number of steps')
parser.add_argument('--config', type=str, nargs=1, help='dataset configuration')

args = parser.parse_args()

def save_checkpoint(state, path, epoch=0, step=0):
    filename = os.path.join(path, 'checkpoint-{}-{}.ckpt.tar'.format(epoch, step))
    torch.save(state, filename)
   
lr = args.lr
n = args.num_steps
N = args.num_epoch
auto_save = args.auto_save
config = json.load(open(args.config))

model = Net()
trainable_var = model.get_trainable()
restore_var = model.get_restorable()
loss = []

for ep in range(N):
    for t in range(n):
        '''
        !TODO!
        x - take from reader
        '''
        y_pred = model(x)

        loss.append(loss_fn(y_pred, y))

        write_summaries({'loss':loss[len(loss)-1]}, config['summary_path'])

        if t % auto_save == 0:
            d = model.get_restorable()
            save_checkpoint(d, config['restore_path'], ep, t)
            print("-----------------------------------------------\n")
            print("step: {}\n losses:\n {}\n".format(t, loss.data[0]))
        
        model.zero_grad()

        loss.backward()
        for param in model.parameters():
            '''
            TODO
            get_trainable()
            '''
            param.data -= learning_rate * param.grad.data


