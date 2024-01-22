import cv2
import numpy as np
import os
# from tqdm import tqdm
from osgeo import gdal
from geo_tools import read_img, write_img


def mosaic(pic_path, pic_target, spatial_reference, RepetitionRate , resultname):
    # 数组保存分割后图片的列数和行数，注意分割后图片的格式为x_x.jpg，x从0开始
    num_width_list = []
    num_lenght_list = []
    picture_names = os.listdir(pic_path)

    # 获取原始图片不带后缀的文件名
    filename = ''
    for root, dirs, files in os.walk(spatial_reference):  # 遍历该文件夹  # 指定文件夹的路径
        for file in files:  # 遍历刚获得的文件名files
            (file_name, extension) = os.path.splitext(file)  # 将文件名拆分为文件名与后缀
            if (extension == '.tif'):  # 判断该后缀是否为.tif文件
                filename = file_name


#    dataset = gdal.Open(spatial_reference + filename + '.tif')
    dataset = gdal.Open(spatial_reference)
    
    minx, xres, xskew, maxy, yskew, yres = dataset.GetGeoTransform()
    proj, geotrans, data = read_img(spatial_reference )  # 读数据
    lon = minx
    lat = maxy

    # 获取分割后图片的尺寸
    
    proj1, geotrans1, data1 = read_img(pic_path + filename + '24_0_0.tif')  # 读数据
#    print(data1.shape)
    depth,width, length = data1.shape

    # 分割名字获得行数和列数，通过数组保存分割后图片的列数和行数
    for picture_name in picture_names:
        num_width_list.append(int(picture_name.split("_")[-2]))
        num_lenght_list.append(int((picture_name.split("_")[-1]).split(".")[0]))
    # 取其中的最大值，从0开始的所以+1
    num_width = max(num_width_list)+1
    num_length = max(num_lenght_list)+1

    if dataset.RasterCount != 1:
        # 预生成拼接后的图片
        splicing_pic = np.zeros((dataset.RasterCount, num_width * (width - RepetitionRate) + RepetitionRate,
                                 num_length * (length - RepetitionRate) + RepetitionRate))
        # 循环复制
        for i in range(0, num_width):
            # print(2)
            for j in range(0, num_length):
                # img_part = cv2.imread(pic_path + '1_{}_{}.tif'.format(i, j), cv2.IMREAD_UNCHANGED)
                prj, geotran, img_part = read_img(pic_path + filename + '_{}_{}.tif'.format(i, j))
                print(pic_path + filename + '_{}_{}.tif'.format(i, j))
                # splicing_pic[width*i : width*(i+1), length*j : length*(j+1)] = img_part
                splicing_pic[0:dataset.RasterCount, (width-RepetitionRate)*i: (width-RepetitionRate)*i+width,
                             (length-RepetitionRate)*j: (length-RepetitionRate)*j+length] = img_part

            # 保存图片，大功告成
            # cv2.imwrite(pic_target + 'result.tif', splicing_pic)
            # print(splicing_pic.shape)
            # 写数据
            write_img(pic_target + resultname, proj, lon, lat, xres, yres, splicing_pic)
    else:
        # 预生成拼接后的图片
        splicing_pic = np.zeros((num_width * (width - RepetitionRate) + RepetitionRate,
                                 num_length * (length - RepetitionRate) + RepetitionRate))
        # 循环复制
        for i in range(0, num_width):
            for j in range(0, num_length):
                # img_part = cv2.imread(pic_path + '1_{}_{}.tif'.format(i, j), cv2.IMREAD_UNCHANGED)
                prj, geotran, img_part = read_img(pic_path + filename + '_{}_{}.tif'.format(i, j))
                print(pic_path + filename + '_{}_{}.tif'.format(i, j))
                # splicing_pic[width*i : width*(i+1), length*j : length*(j+1)] = img_part
                splicing_pic[(width - RepetitionRate) * i: (width - RepetitionRate) * i + width,
                             (length - RepetitionRate) * j: (length - RepetitionRate) * j + length] = img_part

            # 保存图片
            # 写数据
            write_img(pic_target + resultname, proj, lon, lat, xres, yres, splicing_pic)
    print("done!!!")

if __name__ == "__main__":
    # 分割后的图片的文件夹，拼接后要保存的文件夹,原图（获取原图的空间信息并赋值给拼接好的图片）存储文件夹，RepetitionRate（重复率），结果名
    #原图坐标文件名称要与拼接图像一致
    mosaic('E:/NRG/7trainingsamples/0/1/', 'E:/NRG/7trainingsamples/0/', 
           'E:/NRG/7trainingsamples/0/24.tif', 80, 'bc1.tif')











