import random
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.image as mpimg
from tools.geo_tools import read_img,write_img
#from geo_tools import read_img,write_img

#import Image
#import gdal

# 类别对应
#class_values = [100, 200, 300, 400, 500, 600, 700, 800]
class_values = [1, 2]

def normalization(img):
    WFI = img[:, :, 0]
    nir = img[:, :, 1]
    swir = img[:, :, 2]
    WFI=(WFI - 4.5)/15.5
    nir=nir- 0.6
    swir=swir- 0.6
    imgarray = np.dstack((WFI,nir,swir))
    return imgarray



def band_stack(img,nir,swir):
    #img=np.array(img,dtype='int') 
#    img = cv2.bilateralFilter(img,7,75,75)
#    img = img.astype(np.float32)
    np.seterr(divide='ignore',invalid='ignore')
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    R=R/ 127.5 - 1
    G=G/ 127.5 - 1
    B=B/ 127.5 - 1
#    nir=nir/ 127.5 - 1
    evi = evi/1
    swir=swir/ 26
#    imgarray = np.dstack((evi))
#    print(imgarray)
    return imgarray


def index_augument(img):
    #img=np.array(img,dtype='int') 
#    img = cv2.bilateralFilter(img,7,75,75)
    img = img.astype(np.float32)
    np.seterr(divide='ignore',invalid='ignore')
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]    
    R=R/ 127.5 - 1
    G=G/ 127.5 - 1
    B=B/ 127.5 - 1
    
def rotate_bound(img, angle):
    if angle == 90:
        out = Image.fromarray(img).transpose(Image.ROTATE_90)
        return np.array(out)
    if angle == 180:
        out = Image.fromarray(img).transpose(Image.ROTATE_180)
        return np.array(out)
    if angle == 270:
        out = Image.fromarray(img).transpose(Image.ROTATE_270)
        return np.array(out)


def data_augment(x, y):
    flag = random.choice([1, 2, 3, 4, 5, 6])
    if flag == 1:
        x, y = cv2.flip(x, 1), cv2.flip(y, 1)  # Horizontal mirror
    if flag == 2:
        x, y = cv2.flip(x, 0), cv2.flip(y, 0)  # Vertical mirror
    if flag == 3:
        x, y = rotate_bound(x, 90), rotate_bound(y, 90)
    if flag == 4:
        x, y = rotate_bound(x, 180), rotate_bound(y, 180)
    if flag == 5:
        x, y = rotate_bound(x, 270), rotate_bound(y, 270)
    else:
        pass
    return x, y


def get_img_label_paths(evi_path,labels_path):
    res = []
    for dir_entry in os.listdir(evi_path):
        if os.path.isfile(os.path.join(evi_path, dir_entry)):
            file_name, _ = os.path.splitext(dir_entry)
            res.append((os.path.join(evi_path, file_name+".TIF"),
                        os.path.join(labels_path, file_name+".TIF")))
    return res


def get_image_array(img):
    return np.float32(img) / 127.5 - 1


def get_segmentation_array(img, nClasses):
#    print(img.shape)
    assert len(img.shape) == 2
    seg_labels = np.zeros((1024, 1024, nClasses))
#    print(seg_labels)
    for p in class_values:
        img[img == p] = class_values.index(p)

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)
#    print(1)
    seg_labels = np.reshape(seg_labels, (1024, 1024, nClasses))
#    print(2)
    return seg_labels


def train_data_generator(evi_path,labels_path, batch_size, num_class, use_augment):
#    print(1)
    img_seg_pairs = get_img_label_paths(evi_path,labels_path)
#    print(2)
#    print(img_seg_pairs)
#    print(labels_path)
    pairs_number = len(img_seg_pairs)
    while True:
        X, Y = [], []
        random.shuffle(img_seg_pairs)
        for i in range(pairs_number):
#            img = cv2.imread(img_seg_pairs[i][0], cv2.IMREAD_UNCHANGED)     
#            img=get_image_array(img)
            prj,geo,evi=read_img(img_seg_pairs[i][0])
            
            
#            MVI=indices.GetRasterBand(1)
#            NDVI=indices.GetRasterBand(4)
#            MVI = MVI.ReadAsArray() #获取数据
#            NDVI = NDVI.ReadAsArray() #获取数据
            
            evi = np.transpose(evi,(1,2,0))
#            indices=normalization(indices)
#            MVI = indices[:, :, 0]


            seg = cv2.imread(img_seg_pairs[i][1], cv2.IMREAD_UNCHANGED)
#            img=np.dstack((evi))
            
            
#            img=np.dstack((img,MVI,NDVI))
            
#            print(img.shape)
#            if use_augment:
#                img, seg = data_augment(img, seg)
#                nir, seg = data_augment(nir, seg)
#                swir, seg = data_augment(swir, seg)

                        #倪荣光测试
#            img=index_augument(img)
#            X.append(img)
#            print(img)
#            img=band_stack(img,nir,swir)
#            print(img.shape)
            X.append(evi)
#            X.append(get_image_array(img))
            Y.append(get_segmentation_array(seg, num_class))
#            print(3)
            if i == pairs_number - 1:
                yield np.array(X), np.array(Y)
                X, Y = [], []

            if len(X) == batch_size:
                assert len(X) == len(Y)
                yield np.array(X), np.array(Y)
                X, Y = [], []


def val_data_generator(evi_path,labels_path, batch_size, num_class):
    img_seg_pairs = get_img_label_paths(evi_path,labels_path)
    pairs_number = len(img_seg_pairs)

    while True:
        X, Y = [], []
        for i in range(pairs_number):
#            img = cv2.imread(img_seg_pairs[i][0], cv2.IMREAD_UNCHANGED)
#
#            img=get_image_array(img)
            prj,geo,evi=read_img(img_seg_pairs[i][0])
#            print(evi.shape)
            evi = np.transpose(evi,(1,2,0))
            seg = cv2.imread(img_seg_pairs[i][1], cv2.IMREAD_UNCHANGED)
#            img=np.dstack((evi))
#            print(img.shape)
            X.append(evi)
#            X.append(get_image_array(img))
            Y.append(get_segmentation_array(seg, num_class))
            if i == pairs_number - 1:
                yield np.array(X), np.array(Y)
                X, Y = [], []
            if len(X) == batch_size:
                assert len(X) == len(Y)
                yield np.array(X), np.array(Y)
                X, Y = [], []          

def test_data_generator(evi_path, batch_size):
    evi = []
    for dir_entry in os.listdir(evi_path):
        if os.path.isfile(os.path.join(evi_path, dir_entry)):
            assert dir_entry.endswith('tif')
#            print(images_path)
            evi.append(os.path.join(evi_path, dir_entry))
    pairs_number = len(evi) 
    X, R= [], []
    for i in tqdm(range(pairs_number)):
#        print(images[i])
        prj,geo,img=read_img(evi[i])
        img= np.transpose(img,(1,2,0)) 
#        evi=index_augument(img)
        X.append(img)
#        print(images[i])
        R.append(os.path.basename(evi[i]))
        if i == pairs_number - 1:
            yield np.array(X), R
            X, R = [], []
        if len(X) == batch_size:
            yield np.array(X), R
            X, R = [], [] 
          

if __name__ == "__main__":
    data = [[1, 2, 3, 4], [5, 6, 7, 8], [3, 5, 1, 8]]
    arr = np.array(data)
    print(arr.shape, arr)
    print(arr == 1)
    print((arr == 1).astype(int))
    print(arr)
    arr[arr == 1] = 100
    print(arr)
