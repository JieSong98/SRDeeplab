import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import os
import cv2 as cv


array_of_img = [] # this if for store all of the image data
array_of_lab = []
def read_lab_directory(directory_img_name,directory_lab_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_lab_name):

        #img is used to store the image data 
#        img=cv.imread(directory_lab_name + "/" + filename, cv.IMREAD_UNCHANGED)
        print(filename)
#        print(img)
#        print(directory_img_name + "/" + filename)
        img = mpimg.imread(directory_img_name + "/" + filename)
        portion = os.path.splitext(filename)
#        print(portion)
#        if portion[1] == ".TIF":
#           newname = portion[0] + ".TIF"
#           print(newname)
#           print(directory_lab_name + "/" + newname)
#           print(directory_lab_name )
        lab = cv.imread(directory_img_name + "/" + portion[0] + ".TIF", cv.IMREAD_UNCHANGED)
#        print(2)
#        print(lab.shape)
#        print(directory_lab_name + "/" + portion[0] + ".TIF")
        plt.subplot(1,2,1)
        plt.imshow(lab) # 显示图片
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(img) # 显示图片
        plt.axis('off')
        plt.show()
        array_of_img.append(img)
        array_of_lab.append(lab)


read_lab_directory('E:/LXC/ceshi/n1','E:/LXC/ceshi/n2') # 读取和代码处于同一目录下的 lena.png



