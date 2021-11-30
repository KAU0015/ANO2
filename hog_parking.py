import cv2
import numpy as np
import glob

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
IMG_SIZE = 96
#hog = cv2.HOGDescriptor((IMG_SIZE, IMG_SIZE), (32,32), (16,16), (8,8), 9, 1, -1, 0, 0.2, 1, 64, True)
hog = cv2.HOGDescriptor((IMG_SIZE, IMG_SIZE), (64, 64), (32, 32), (16, 16), 9, 1, -1, 0, 0.09, 1, 64, True)
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_INTER)
svm.setC(100.0)

svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

for i in range(len(train_images_full)):
    one_park_image = cv2.imread(train_images_full[i], 0)
    res_image = cv2.resize(one_park_image, (IMG_SIZE, IMG_SIZE))
    hog_feature = hog.compute(res_image)
    train_images_list.append(hog_feature)
    train_labels_list.append(1)

for i in range(len(train_images_free)):
    one_park_image = cv2.imread(train_images_free[i], 0)
    res_image = cv2.resize(one_park_image, (IMG_SIZE, IMG_SIZE))
    hog_feature = hog.compute(res_image)
    train_images_list.append(hog_feature)
    train_labels_list.append(0)

print("train_all ", len(train_images_list))


svm.train(np.array(train_images_list), cv2.ml.ROW_SAMPLE, np.array(train_labels_list))
#svm.save("my_hog.xml")

print("TRAIN HOG DONE")

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
        gray_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2GRAY)
    

        hog_feature = hog.compute(gray_image)
        predict_label = svm.predict(np.array(hog_feature).reshape(1, -1))
        predict_label = predict_label[1][0][0]
        #print(predict_label)
    

        if(predict_label == 0):
            result_list.append(0)
            cv2.line(one_park_image, (int(one_c[0]), int(one_c[1])), (int(one_c[2]), int(one_c[3])), (255,0,0), 2)
            cv2.line(one_park_image, (int(one_c[2]), int(one_c[3])), (int(one_c[4]), int(one_c[5])), (255,0,0), 2)
            cv2.line(one_park_image, (int(one_c[4]), int(one_c[5])), (int(one_c[6]), int(one_c[7])), (255,0,0), 2)
            cv2.line(one_park_image, (int(one_c[6]), int(one_c[7])), (int(one_c[0]), int(one_c[1])), (255,0,0), 2)
        else:
            result_list.append(1)
            cv2.line(one_park_image, (int(one_c[0]), int(one_c[1])), (int(one_c[2]), int(one_c[3])), (0,0,255), 2)
            cv2.line(one_park_image, (int(one_c[2]), int(one_c[3])), (int(one_c[4]), int(one_c[5])), (0,0,255), 2)
            cv2.line(one_park_image, (int(one_c[4]), int(one_c[5])), (int(one_c[6]), int(one_c[7])), (0,0,255), 2)
            cv2.line(one_park_image, (int(one_c[6]), int(one_c[7])), (int(one_c[0]), int(one_c[1])), (0,0,255), 2)

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
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2.0*((precision*recall)/(precision+recall))

print("acc: ", acc)
print("precision: ", precision)
print("recall: ", recall)
print("F1: ", f1)
