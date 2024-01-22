from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os, sys
from tqdm import tqdm
from tools.dataloader import test_data_generator
import config
import tensorflow.keras.backend as backend
from tools.layers import GlobalAveragePooling2D
import gdal,osr
from tools.geo_tools import read_img,write_img
#from Imags_mosaic import mosaic
import tensorflow.keras.backend as K
  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = config.Config('test')

#定义预测文件的坐标系
def assign_spatial_reference_byfile(src_path, dst_path):
    for filename in os.listdir(src_path):
        src_ds = gdal.Open(src_path + "/" + filename, gdal.GA_ReadOnly)
        sr = osr.SpatialReference()
        sr.ImportFromWkt(src_ds.GetProjectionRef())
        geoTransform = src_ds.GetGeoTransform()
        dst_ds = gdal.Open(dst_path + "/" + filename, gdal.GA_Update)
        dst_ds.SetProjection(sr.ExportToWkt())
        dst_ds.SetGeoTransform(geoTransform)
        dst_ds = None
        src_ds = None

def predict_single(evi_path, output_path, model, n_class):

    for image in tqdm(os.listdir(input_path)):
        index, _ = os.path.splitext(image)
        img = cv2.imread(os.path.join(input_path, image), cv2.IMREAD_UNCHANGED)
        img = np.float32(img) / 127.5 - 1
        pr = model.predict(np.expand_dims(img, axis=0), verbose=1)[0]
        pr = pr.reshape((1024, 1024, n_class)).argmax(axis=2)
        seg_img = np.zeros((1024, 1024), dtype=np.uint16)
        for c in range(n_class):
            seg_img[pr[:, :] == c] = int((c + 1) * 100)
        cv2.imwrite(os.path.join(output_path, index + ".png"), seg_img)


def predict_batch(input_path, output_path, model, n_class, spatial_reference):
    g = test_data_generator(input_path, cfg.batch_size)
    num_width_list = []
    num_lenght_list = []
    picture_names = os.listdir(input_path)
    for picture_name in picture_names:
        num_width_list.append(int(picture_name.split("_")[1]))
        num_lenght_list.append(int((picture_name.split("_")[-1]).split(".")[0]))

    (width, length) = [1024,1024]
    (width1, length1) = [64,64]

    dataset = gdal.Open(spatial_reference)
    minx, xres, xskew, maxy, yskew, yres = dataset.GetGeoTransform()
    xres = xres/16
    yres = yres/16
    proj = dataset.GetProjection()          # 只读坐标数据
    widthz = dataset.RasterXSize  # 栅格矩阵的列数(行)
    heightz = dataset.RasterYSize  # 栅格矩阵的行数(列) 为了与下面对于应
    # bands = dataset.RasterCount
    del dataset

    lon=minx+xres
    lat=maxy+yres

    datatype = gdal.GDT_Byte
    z_width = (widthz + (width1 - widthz % width1))*16
    z_height = (heightz + (length1 - heightz % length1))*16
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    # 输出文件名字
    dataset2 = driver.Create(output_path + '/' + '0.tif', z_width, z_height, 1, datatype, options=["BIGTIFF=YES", "COMPRESS=LZW"])
    dataset2.SetGeoTransform((lon, xres, 0, lat, 0, yres))  # 写入仿射变换参数
    dataset2.SetProjection(proj)  # 写入投影
    #del dataset2
    
    f = 0
    for x, r in g:
        out = model.predict(x, verbose=0)
        # out = model(x, training=False)
        for i in range(out.shape[0]):
            pr = out[i].reshape((1024, 1024, n_class)).argmax(axis=2)
            seg_img = np.zeros((1024, 1024), dtype=np.uint8)
            for c in range(n_class):
                seg_img[pr[:, :] == c] = int((c + 1) )

            dataset2.GetRasterBand(1).WriteArray(seg_img, length*(int((r[i].split("_")[-1]).split(".")[0])), width*(int(r[i].split("_")[1]))) 
            f = f + 1
            if f%1000==0:
                dataset2.FlushCache()         
    del dataset2

def myloss(y_true, y_pred):
    return K.mean(-y_true*K.log(y_pred + K.epsilon())*0.8-(1-y_true)*K.log(1-y_true + K.epsilon()), axis=-1)    

if __name__ == "__main__":
    weights_path = cfg.weight_path
    input_path = cfg.evi_path
    output_path = cfg.output_path
    n_class = cfg.n_classes
    cfg.check_folder(output_path)
    originimage_path=r''
#    原图的位置，用于获取坐标系
    
    if weights_path is None:
        print('weights_path  ERROR!')
        sys.exit()
    print(f'loaded : {weights_path}')
    model = load_model(weights_path, custom_objects= {'GlobalAveragePooling2D': GlobalAveragePooling2D,'backend':backend})
    # model = load_model(weights_path, custom_objects= {'myloss': myloss, 'GlobalAveragePooling2D' : GlobalAveragePooling2D,'backend':backend})
    predict_batch(input_path, output_path, model, n_class,originimage_path)
    
    
    