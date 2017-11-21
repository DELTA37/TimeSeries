import moex.collecter
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('security', type=str)
parser.add_argument('--path', type=str)
parser.add_argument('--splitit', type=int)
parser.add_argument('--save', type=str)

args = parser.parse_args()

security = args.security
splitit = args.splitit
save = args.save

if security == 'list':
    print(moex.collecter.getSecurityList()[0].secid.to_string())
    exit()

if splitit == None:
    splitit = 0

if args.path == None:
    path = './'
else:
    path = args.path

if save is not None:
    arr = np.array(moex.collecter.getPriceArray(security)['CLOSE'], dtype=np.float32)
    np.save(save, arr)
else:    
    moex.collecter.PlotData(security, path, splitit)
