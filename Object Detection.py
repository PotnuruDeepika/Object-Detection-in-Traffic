import os
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly
import cv2
from glob import glob
l=[]
#loading img
#path=glob('D:/DEEPIKA/TMG/Videos/traffic1/*.png')
#for img in path:
imagePath='D:/DEEPIKA/TMG/Videos/traffic1/0.png'
img = cv2.imread(imagePath)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.axis("off")
plt.imshow(img1)
plt.show()

    #creating boxes
box,label,count=cv.detect_common_objects(img)
output=draw_bbox(img,box,label,count)

output=cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.axis("off")
plt.imshow(output)
plt.show()

    #count object in the img
print("Number of object:",len(label))
#     l.append((len(label)))
# print('Max object:',max(l))


