import os
from osgeo import gdal
from geo_tools import read_img,write_img
from metrics import f1_score
import tensorflow as tf
import numpy as np

def precision(Tpath, Ppath):
    i = 0
    MPrecision = 0 
    MRecall = 0
    MF1 = 0
    
    for file_name in os.listdir(Tpath):
        portion = os.path.splitext(file_name)
        
        if portion[1] == ".tif":
            # Tdataset = gdal.Open(Tpath + "/" + file_name)
            # Pdataset = gdal.Open(Ppath + "/" + file_name)
            
            proj1, geotrans1, data1 = read_img(Tpath + "/" + file_name)  # 读数据
            proj2, geotrans2, data2 = read_img(Ppath + "/" + file_name)  # 读数据            
            
            T = tf.cast(data1, dtype=tf.float32)
            P = tf.cast(data2, dtype=tf.float32)
            
            # 1代表正确区域
#            
#            T = np.where(T == 1, 0, T)
#            T = np.where(T == 2, 1, T)
#            
#            P = np.where(P == 1, 0, P)
#            P = np.where(P == 2, 1, P)
                               
            precision, recall,  f1 = f1_score(T, P)
            
            Precision = precision.numpy()
            Recall = recall.numpy()
            F1 = f1.numpy()
            
            MPrecision += Precision
            MRecall += Recall
            MF1 += F1
                      
            if Precision != 0:
                i = i + 1
            
    MPrecision = MPrecision/i
    MRecall = MRecall/i
    MF1 = MF1/i              
      
    print("MPrecision：", MPrecision)
    print("MRecall：", MRecall)
    print("MF1：", MF1)
     
        

if __name__ == "__main__":
    
#    precision(r"E:\LXC\DeepTrain\data\Truth", r"E:\LXC\DeepTrain\data\prediction")
    precision(r"F:\YCY2\chutu\compute_precision\2020_nrg_label", r"F:\YCY2\chutu\compute_precision\my_result_rasters")
    
    
