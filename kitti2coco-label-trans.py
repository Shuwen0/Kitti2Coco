import numpy as np
import cv2
import os
import sys
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json

coco = Coco()

#upper directory
current_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

#data_set path (relative)
kitti_img_path =  current_dir + '/' + 'RawImages/training/image_2/'
kitti_label_path = current_dir + '/' + 'RawLabels/training/label_2/'


#transformed lables path (relative)
#kitti_label_tosave_path = 'kitti/labels2coco/'
kitti_label_tosave_path = current_dir + '/' + 'TransLabels/'

#the real ptah of your data set (relative)
kitti_data_real_path = current_dir + '/' + 'RawImages/training/image_2/'

index = 0
cvfont = cv2.FONT_HERSHEY_SIMPLEX


kitti_names = open('kitti.names','r')
kitti_names_contents = kitti_names.readlines() 

# add categories
for i in range(len(kitti_names_contents)):
    coco.add_category(CocoCategory(id=i, name=kitti_names_contents[i]))

#debugging
#print("coco category")
#print(coco.category_)#no idea how to check category

#filenames of images and labels（without dir）
kitti_images = os.listdir(kitti_img_path)
kitti_labels = os.listdir(kitti_label_path)

#make sure there are equal number of images and labels
assert len(kitti_images) == len(kitti_labels)

#match
kitti_images.sort()
kitti_labels.sort()

kitti_names_dic_key = []

#[car, van, ...]
for class_name in kitti_names_contents:
    kitti_names_dic_key.append(class_name.rstrip())

#[0,1,...,7]
values = range(len(kitti_names_dic_key))

#{car:0; ...}
kitti_names_num = dict(zip(kitti_names_dic_key,values))

print("kitti_names_num")
print(kitti_names_num)

#record all train image
# mode="w" automatically creates new file, or delete previous contents
f = open('trainvalno5k-kitti.txt','w')
for img in kitti_images:
    #absolute path to train iamge
    f.write(kitti_data_real_path+img+'\n')
f.close()


kitti_images_with_labels = []
for indexi in range(len(kitti_images)):
    
    #abosolute path to each train image
    kitti_img_totest_path = kitti_img_path + kitti_images[indexi]
    #abosolute path to each label
    kitti_label_totest_path = kitti_label_path + kitti_labels[indexi]

    #make sure they match
    assert kitti_images[indexi][0:-4] == kitti_labels[indexi][0:-4]

    #print("kitti label and image")
    #print(kitti_label_totest_path,kitti_img_totest_path)
    
    #read image
    kitti_img_totest = cv2.imread(kitti_img_totest_path)
    
    #get hight & width
    img_height, img_width = kitti_img_totest.shape[0],kitti_img_totest.shape[1]
    
    #create coco image:
    coco_image = CocoImage(file_name=kitti_images[indexi], height=img_height, width=img_width)
    
    
    #读取这一张图片相应的label（有些img没有标注，跳过）
    #try:
    #    kitti_label_totest = open(kitti_label_totest_path,'r')
    #    #如果能够读取label，证明这张图片有标注，记录img路径
    #    kitti_images_with_labels.append(kitti_img_totest_path)
    #except:
    #    continue
        
    #target per line
    label_contents = kitti_label_totest.readlines()
    #print(label_contents)
    real_label = open(kitti_label_tosave_path + kitti_labels[indexi],'w')
    
    for line in label_contents:
        data = line.split(' ')
        x=y=w=h=0
        if(len(data) == 15):
            
            #class name is the first item
            class_str = data[0]
            if(class_str != 'DontCare'):
                # for kitti calls is a string
                # trans this to number by using kitti.names
                #(x,y) center (w,h) size
                x1 = float(data[4])
                y1 = float(data[5])
                x2 = float(data[6])
                y2 = float(data[7])
                
                intx1 = int(x1)
                inty1 = int(y1)
                intx2 = int(x2)
                inty2 = int(y2)

                bbox_center_x = float( (x1 + (x2 - x1) / 2.0) / img_width)
                bbox_center_y = float( (y1 + (y2 - y1) / 2.0) / img_height)
                bbox_width = float((x2 - x1) / img_width)
                bbox_height = float((y2 - y1) / img_height)
                
                #加入annotations:
                coco_image.add_annotation(
                    CocoAnnotation(
                    bbox=[bbox_center_x, bbox_center_y, bbox_width, bbox_height],
                    category_id=kitti_names_num[data[0]],
                    category_name=class_str
                    )
                )

                #print(kitti_names_contents[class_num])
                # cv2.putText()
                # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
                #cv2.putText(kitti_img_totest, class_str, (intx1, inty1+3), cvfont, 2, (0,0,255), 1)
                # cv2.rectangle()
                # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
                #cv2.rectangle(kitti_img_totest, (intx1,inty1), (intx2,inty2), (0,255,0), 2)
                #line_to_write = str(kitti_names_num[class_str]) + ' ' + str(bbox_center_x)+ ' ' + str(bbox_center_y)+ ' ' + str(bbox_width)+ ' ' + str(bbox_height) +'\n'
                #real_label.write(line_to_write)
                sys.stdout.write(str(int((indexi/len(kitti_images))*100))+'% '+'*******************->' "\r" )
                sys.stdout.flush()

    #cv2.imshow(str(indexi)+' kitti_label_show',kitti_img_totest)    
    #cv2.waitKey()
    
    #add annotation to coco
    coco.add_image(coco_image)
    #real_label.close()
    
save_path = (current_dir + '/' + 'annotations/instances_train2017.json')
save_path = (current_dir + '/' + 'annotations/instances_val2017.json')
save_json(data=coco.json, save_path=save_path)
kitti_names.close()

#record the path to all images with label
f = open('kitti_img_with_labels.txt','w')
for path in kitti_images_with_labels:
    f.write(path+'\n')
    
    
    
print("Labels tranfrom finished!")
