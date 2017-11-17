import moex.collecter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('security', type=str)
parser.add_argument('--path', type=str)
parser.add_argument('--splitit', type=int)

args = parser.parse_args()

security = args.security
splitit = args.splitit

if security == 'list':
    print(moex.collecter.getSecurityList()[0].secid.to_string())
    exit()
if splitit == None:
    splitit = 0

if args.path == None:
    path = './'
else:
    path = args.path

moex.collecter.PlotData(security, path, splitit)
