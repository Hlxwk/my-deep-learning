import tensorflow as tf
import numpy as np
import xlrd 
import matplotlib.pyplot as plt
import os
from sklearn.utils import check_random_state
import sys
import h5py
sys.path.append('/home/hanwenkai/桌面/pcl_code2/hdf5_python')
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
'''
base_dir=os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data')
www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
zipfile = os.path.basename(www)    # return final filename
#print(data_path)
print(zipfile[:-4])
gg = 'build'
os.system('rm -r %s'% (gg))
'''
'''
h5_path = sys.argv[1]
f = h5py.File(h5_path,'r')
point_data = f['data'][:]
#print(point_data)
#print(point_data.shape)#2048,2048,3
shape_pc = point_data[1,...]
print(shape_pc)
print(shape_pc.shape)
print(shape_pc.reshape((-1,3)).shape)
'''

ply_path = sys.argv[1]
plydata = PlyData.read(ply_path)
pc = plydata['vertex'].data
print(pc)
print(pc.shape)
pc_array = np.array([[c[0],c[1],c[2]] for c in pc])
print(pc_array.shape)
