# import libraries
import os
import numpy as np
import pandas as pd
import cv2
import gc
from tqdm import tqdm
from glob import glob

# Step-1 and 2
#   + collect all data
#   + labeling
dirs = os.listdir('dataset')
images_path = []
labels = []
for folder in dirs:
    path = glob('./dataset/{}/*.jpg'.format(folder))
    label =['{}'.format(folder)]*len(path)
    # append
    images_path.extend(path)
    labels.extend(label)

# Step-3 & 4
#   + Face Detection
#   + Cropping
img_path = images_path[1]
img = cv2.imread(img_path)

cv2.imshow('original', img)
cv2.waitKey()
cv2.destroyAllWindows()
# face detection
face_detection_model = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt',
                                                './models/res10_300x300_ssd_iter_140000_fp16.caffemodel')


def face_detection_dnn(img):
    # blob from image (rgb mean subraction image)
    image = img.copy()
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1, (300, 300), (104, 117, 123), swapRB=True)
    # get the detections
    face_detection_model.setInput(blob)
    detections = face_detection_model.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # confidence score
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype(int)
            # print(box)
            pt1 = (box[0], box[1])
            pt2 = (box[2], box[3])
            # cv2.rectangle(image,pt1,pt2,(0,255,0),2)
            roi = image[box[1]:box[3], box[0]:box[2]]

            return roi
    return None

img_roi = face_detection_dnn(img)

cv2.imshow('roi', img_roi)
cv2.imshow('original', img)
cv2.waitKey()
cv2.destroyAllWindows()

# Step 5:
#   + Blob from image

def datapreprocess(img):
    # blob from image (rgb mean subtraction image)
    face = face_detection_dnn(img)
    if face is not None:

        # computing blob from image
        blob = cv2.dnn.blobFromImage(face, 1, (100, 100), (104, 117, 123), swapRB=True)
        blob_squeeze = np.squeeze(blob).T
        blob_rotate = cv2.rotate(blob_squeeze, cv2.ROTATE_90_CLOCKWISE)
        blob_flip = cv2.flip(blob_rotate, 1)
        # remove negative values and normalize
        img_norm = np.maximum(blob_flip, 0) / blob_flip.max()

        return img_norm
    else:
        return None

# Apply to all Image and Append in a List

# len(images_path)

data_img = []
label_img = []
i = 0
for path, label in tqdm(zip(images_path, labels), desc='preprocessing'):
    img = cv2.imread(path)
    process_img = datapreprocess(img)
    if process_img is not None:
        data_img.append(process_img)
        label_img.append(label)

    i += 1
    if i % 100 == 0:
        gc.collect()


X = np.array(data_img)
y = np.array(label_img)

X.shape, y.shape

np.savez('./dataset/data_preprocess.npz', X, y)