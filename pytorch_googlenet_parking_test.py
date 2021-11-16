import cv2
import numpy as np
import glob
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import os
from PIL import Image

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p1_2 = nn.BatchNorm2d(c1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.BatchNorm2d(c2[0])
        self.p2_3 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p2_4 = nn.BatchNorm2d(c2[1])
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.BatchNorm2d(c3[0])
        self.p3_3 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p3_4 = nn.BatchNorm2d(c3[1])
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        self.p4_3 = nn.BatchNorm2d(c4)

    def forward(self, x):
        p1 = F.relu(self.p1_2(self.p1_1(x)))
        p2 = F.relu(self.p2_3(self.p2_2(F.relu(self.p2_2(self.p2_1(x))))))
        p3 = F.relu(self.p3_3(self.p3_2(F.relu(self.p3_2(self.p3_1(x))))))
        p4 = F.relu(self.p4_3(self.p4_2(self.p4_1(x))))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(96)
    
])

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

#cv2.namedWindow("detection", 0)
pkm_file = open("parking_map_python.txt", "r")
pkm_lines = pkm_file.readlines()
pkm_coordinates = []


for line in pkm_lines:
    st_line = line.strip()
    sp_line = list(st_line.split(" "))
    pkm_coordinates.append(sp_line)

train_images_full = [img for img in glob.glob("train_images/full/*.png")]
train_images_free = [img for img in glob.glob("train_images/free/*.png")]

train_labels_list = []
train_images_list = []
train_images_list_lbp = []
IMG_SIZE = 96


net = torch.load("my_googlenet.pth")

#svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

for i in range(len(train_images_full)):
    one_park_image = cv2.imread(train_images_full[i], 0)
    res_image = cv2.resize(one_park_image, (IMG_SIZE, IMG_SIZE))
   # hog_feature = hog.compute(res_image)
  #  train_images_list.append(hog_feature)
    train_labels_list.append(1)
    train_images_list_lbp.append(res_image)

for i in range(len(train_images_free)):
    one_park_image = cv2.imread(train_images_free[i], 0)
    res_image = cv2.resize(one_park_image, (IMG_SIZE, IMG_SIZE))
  #  hog_feature = hog.compute(res_image)
  #  train_images_list.append(hog_feature)
    train_labels_list.append(0)
    train_images_list_lbp.append(res_image)


test_images = [img for img in glob.glob("test_images/*.jpg")]
test_images.sort()
result_list = []

for img in test_images:
    one_park_image = cv2.imread(img)
    one_img_paint =one_park_image.copy()

    for one_c in pkm_coordinates:
        pts = [((float(one_c[0])), float(one_c[1])),
                ((float(one_c[2])), float(one_c[3])),
                ((float(one_c[4])), float(one_c[5])),
                ((float(one_c[6])), float(one_c[7]))] 
        #print(pts)
        #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        warped_image = four_point_transform(one_park_image, np.array(pts))
        res_image = cv2.resize(warped_image, (IMG_SIZE, IMG_SIZE)) 
        one_img_rgb = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(one_img_rgb)
        image_pytorch = transform(img_pil) #.to(device)
        image_pytorch = image_pytorch.unsqueeze(0)
        output_pytorch = net(image_pytorch)
        _, predicted = torch.max(output_pytorch, 1)
        #print(predicted)

        #hog_feature = hog.compute(gray_image)
    #    predict_label = svm.predict(np.array(hog_feature).reshape(1, -1))
     #   predict_label = predict_label[1][0][0]
        #print(predict_label)
    
       # predict_label_lbp, predict_confidence_lbp = LBP_recognizer.predict(gray_image)
      #  print(predict_label_lbp)

        if(predicted == 1):
            result_list.append(1)
            cv2.line(one_park_image, (int(one_c[0]), int(one_c[1])), (int(one_c[2]), int(one_c[3])), (0,0,255), 2)
            cv2.line(one_park_image, (int(one_c[2]), int(one_c[3])), (int(one_c[4]), int(one_c[5])), (0,0,255), 2)
            cv2.line(one_park_image, (int(one_c[4]), int(one_c[5])), (int(one_c[6]), int(one_c[7])), (0,0,255), 2)
            cv2.line(one_park_image, (int(one_c[6]), int(one_c[7])), (int(one_c[0]), int(one_c[1])), (0,0,255), 2)
        else:
            result_list.append(0)
            cv2.line(one_park_image, (int(one_c[0]), int(one_c[1])), (int(one_c[2]), int(one_c[3])), (0,255,0), 2)
            cv2.line(one_park_image, (int(one_c[2]), int(one_c[3])), (int(one_c[4]), int(one_c[5])), (0,255,0), 2)
            cv2.line(one_park_image, (int(one_c[4]), int(one_c[5])), (int(one_c[6]), int(one_c[7])), (0,255,0), 2)
            cv2.line(one_park_image, (int(one_c[6]), int(one_c[7])), (int(one_c[0]), int(one_c[1])), (0,255,0), 2)

   # cv2.imshow('one_park_image', one_park_image)
 #   key = cv2.waitKey(0)
 #   if key == 27: # exit on ESC
 #       break

gt_file = open("groundtruth.txt", "r")
gt_lines = gt_file.readlines()
false_positives = 0
false_negatives = 0
true_positives = 0
true_negatives = 0
i = 0

for line in gt_lines:
    gt=int(line.strip())
    if((result_list[i]==1) and (gt==0)):
        false_positives += 1
    if((result_list[i]==1) and (gt==1)):
        true_positives += 1
    if((result_list[i]==0) and (gt==0)):
        true_negatives += 1
    if((result_list[i]==0) and (gt==1)):
        false_negatives += 1
    i += 1

print("true positives: ", true_positives)
print("true negatives: ", true_negatives)
print("false negatives: ", false_negatives)
print("false positives: ", false_positives)
acc = (true_positives + true_negatives)/(false_positives+false_negatives+true_negatives+true_positives)
precision = true_positives / float(true_positives + false_positives+0.0000000001)
recall = float(true_positives) / float(true_positives + false_negatives+0.0000000001)
f1 = 2.0*((precision*recall)/(precision+recall))

print("acc: ", acc)
print("precision: ", precision)
print("recall: ", recall)
print("F1: ", f1)
