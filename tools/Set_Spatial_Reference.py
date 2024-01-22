# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:18:30 2021

@author: Administrator
"""
import os
import gdal
import osr

def assign_spatial_reference_byfile(src_path, dst_path):
    for filename in os.listdir(src_path):
        src_ds = gdal.Open(src_path + "/" + filename, gdal.GA_ReadOnly)
        sr = osr.SpatialReference()
        sr.ImportFromWkt(src_ds.GetProjectionRef())
        geoTransform = src_ds.GetGeoTransform()
        dst_ds = gdal.Open(dst_path + "/" + filename, gdal.GA_Update)
        print(dst_path + "/" + filename)
        dst_ds.SetProjection(sr.ExportToWkt())
        dst_ds.SetGeoTransform(geoTransform)
        dst_ds = None
        src_ds = None
        
        
if __name__=='__main__':
    assign_spatial_reference_byfile('E:/NRG/guangdongguangxiSplit/split/1/RGB','E:/NRG/guangdongguangxiPredictedLabel/1无坐标系')