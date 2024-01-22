import os
from osgeo import gdal
from geo_tools import read_img,write_img

def split(inputpath,outpath):
    for file_name in os.listdir(inputpath):
        portion = os.path.splitext(file_name)
        size=512
        if portion[1] == ".tif":
            dataset = gdal.Open(inputpath + "/" + file_name)    
            print(inputpath + "/" + file_name)
            minx, xres, xskew, maxy, yskew, yres = dataset.GetGeoTransform()
            proj, geotrans, data = read_img(inputpath + "/" + file_name)  # 读数据
#            print(data.shape)
        #        if len(data.shape) != 2:
            if dataset.RasterCount != 1:
        #            depth,width,height = data.shape
                depth,width,height = data.shape
#                size=256
                for j in range(height // size):  # 切割成256*256小图     
                    for i in range(width // size):
        #                    cur_image = data[0:3,i * size:(i + 1) * size, j * size:(j + 1) * size]
                        cur_image = data[0:dataset.RasterCount,i * size:(i + 1) * size, j * size:(j + 1) * size]
                        lon=minx+xres*size*j
                        lat=maxy+yres*(i*size)
    #                    portion = os.path.splitext(file_name)
                        write_img(outpath +'{}_{}_{}.tif'.format(portion[0],i, j), proj,
                             lon, lat, xres, yres, cur_image)  ##写数据
                        print(outpath+'{}_{}_{}.tif'.format(portion[0],i, j))
            else:
                width,height = data.shape
#                size=256
                for j in range(height // size):  # 切割成256*256小图     
                    for i in range(width // size):         
                        cur_image = data[i * size:(i + 1) * size, j * size:(j + 1) * size]
                        lon=minx+xres*size*j
                        lat=maxy+yres*(i*size)
    #                    portion = os.path.splitext(file_name)
                        write_img(outpath+'{}_{}_{}.tif'.format(portion[0],i, j), proj,
                             lon, lat, xres, yres, cur_image)  ##写数据
                        print(outpath+'{}_{}_{}.tif'.format(portion[0],i, j))
    

if __name__ == "__main__":
    # os.chdir(r'E:/data')  # 切换路径到待处理图像所在文件夹
#    E:\NRG\The fifth set of training samples\rgb
#    split('E:/LXC/DeepLabV3+/data/8Index_composite/8Index_composite_predict/RGB','E:/LXC/DeepLabV3+/data/8Index_composite/8Index_composite_predict/splitRGB/')  
#    split('E:/LXC/BCResample/test','E:/LXC/DeepLabV3+/data/predictData/Plain/split_RGB/')

#    split('E:/LXC/GEE_Phenology_Raw/Plain_data/Plain0.8m/nine part','E:/LXC/DeepLabV3+/data/predictData/Plain/split_RGB2/')
    
#    split('E:/LXC/GEE_Phenology_Raw/Plain_data/projection/new/fourpart/RGB','E:/LXC/DeepLabV3+/data/predictData/Plain/projection/splitRGB/')    
#    split('E:/LXC/GEE_Phenology_Raw/Plain_data/projection/new/fourpart/8Index','E:/LXC/DeepLabV3+/data/predictData/Plain/projection/split8Index/')

#    split('E:/LXC/DeeplabV3+/train_8index_composite/Plain/reprediction/RGB/re7','E:/LXC/DeeplabV3+/train_8index_composite/Plain/reprediction/RGB/split7/')
#    split('E:/LXC/DeeplabV3+/train_8index_composite/Plain/reprediction/8Index/re6','E:/LXC/DeeplabV3+/train_8index_composite/Plain/reprediction/8Index/split6/')
    split('E:/LXC/Water/S2imageResample/shandong1','E:/LXC/Water/S2imageResample/shandong1/split/')

#    split('E:/LXC/GEE_Phenology_Raw/PlainSplit/GErgb0.8m/','E:/LXC/DeepLabV3+/data/predictData/Plain/split_RGB2/')    

#    split('E:/LXC/GEE_Phenology_Raw/8bands_clip_to_train/','E:/LXC/GEE_Phenology_Raw/8bands_clip_to_train/split/')     
    #左边输入待分割文件的路径，右边分割结果路径,右边下划线一定要加
    
