# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 20:39:23 2022

@author: Administrator
"""

import numpy as np
from PIL import Image
import os
from osgeo import gdal
from geo_tools import read_img,write_img

# 判断是否需要进行图像填充
def judge(width,height,size,bands,datatype,arraydata):
    new_width = width 
    new_height = height
    
    if width % size != 0:
        new_width = (width//size + 1) * size
    if height % size != 0:
        new_height = (height//size + 1) * size
    # 新建一个新尺寸的矩阵
    if bands != 1:
        new_array = np.zeros((bands,new_width,new_height),dtype=datatype)
        new_array[0:bands,0:width,0:height] = arraydata#这里是把原来的数据复制到新矩形的左上角。
        return new_array,new_width,new_height
    else:
        new_array = np.zeros((new_width,new_height),dtype=datatype)
        new_array[0:width,0:height] = arraydata
        return new_array,new_width,new_height

def main():
    for file_name in os.listdir(input_path):#得到路径 path 下 的 所有文件，返回list列表形式
        portion = os.path.splitext(file_name)#获取去除后缀的文件名。
        if portion[1] == ".tif":
            dataset = gdal.Open(input_path + "/" + file_name)
            minx, xres, xskew, maxy, yskew, yres = dataset.GetGeoTransform()
            proj, geotrans, data = read_img(input_path + "/" + file_name)  # 读数据
            if dataset.RasterCount != 1:
                depth,width,height = data.shape
                size=256
                new_data,new_width,new_height = judge(width,height,size,dataset.RasterCount,data.dtype,data)
                for j in range(new_height // size):  # 切割成256*256小图 
                    for i in range(new_width // size):
                        cur_image = new_data[0:dataset.RasterCount,i * size:(i + 1) * size, j * size:(j + 1) * size]
                        lon=minx+xres*size*j
                        lat=maxy+yres*(i*size)
                        write_img(output_path +'{}_{}_{}.tif'.format(portion[0],i, j), proj,
                             lon, lat, xres, yres, cur_image)  ##写数据
                        print(output_path+'{}_{}_{}.tif'.format(portion[0],i, j))
            else:
                width,height = data.shape
                size=256
                new_data,new_width,new_height = judge(width,height,size,dataset.RasterCount,data.dtype,data)
                for j in range(new_height // size):
                    for i in range(new_width // size):
                        cur_image = new_data[i * size:(i + 1) * size, j * size:(j + 1) * size]
                        lon=minx+xres*size*j
                        lat=maxy+yres*(i*size)
                        write_img(output_path+'{}_{}_{}.tif'.format(portion[0],i, j), proj,
                             lon, lat, xres, yres, cur_image)  ##写数据
                        print(output_path+'{}_{}_{}.tif'.format(portion[0],i, j))


if __name__ == '__main__':
    input_path ="E:/LXC/Water/Label/S2Label/tiff"
    output_path = "E:/LXC/Water/Label/S2Label/tiff/split2/"
    #左边输入待分割文件的路径，右边分割结果路径,右边下划线一定要加
    main()
