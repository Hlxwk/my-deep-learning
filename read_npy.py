import tensorflow as tf
import numpy as np
import os
import sys
import argparse

cur_file_path = os.path.abspath(__file__)
hpy_default_path = os.path.dirname(cur_file_path)  #father path
hpy_default_path += '/Alex-net/AlexNet_with_tensorflow/bvlc_alexnet.npy'
#print(hpy_default_path)

parser = argparse.ArgumentParser(description = 'read hpy')
parser.add_argument('-f', '--path', default=hpy_default_path)
args = parser.parse_args(sys.argv[1:])


wDict = np.load(args.path, encoding = 'bytes').item()
for name in wDict:
    print(name)
    for p in wDict[name]:
        print(p.shape)




