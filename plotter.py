import moex.collecter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('security', type=str)
parser.add_argument('--path', type=str)

args = parser.parse_args()

security = args.security
if args.path == None:
    path = './'
else:
    path = args.path

moex.collecter.PlotData(security, path)
