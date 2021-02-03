"""
Humble YOLO implementation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

grid_w = 2
grid_h = 2
cell_w = 32
cell_h = 32
num_bboxes = 1
num_images = 5000
img_w = grid_w * cell_w
img_h = grid_h * cell_h 


def load_image(j):
    img = cv2.imread('Images/%d.PNG' % j)
    x_t = img_to_array(img)
    with open("Labels/%d.txt" % j, "r") as f:
        y_t = []
        for row in range(grid_w):
            for col in range(grid_h):
                c_t = [float(i) for i in f.readline().split()]
                [x, y, w, h] = [float(i) for i in f.readline().split()]        
                conf_t = [float(i) for i in f.readline().split()]                
                elt = []
                elt += c_t
                for b in range(num_bboxes):
                    elt += [x/cell_w, y/cell_h, w/img_w, h/img_h] + conf_t
                y_t.append(elt)
        assert(f.readline()=="---\n")       
    return [x_t, y_t]


def build_dataset(low_bound=25, up_bound=num_images):
    x_train, y_train = [], []
    for x in range(low_bound, up_bound):
        [x, y] = load_image(x)
        x_train.append(x)
        y_train.append(y)
    return np.array(x_train), np.array(y_train)


def yolo_loss(y_true, y_pred, grid_h=grid_h, grid_w=grid_w):
    grid = np.array([[[float(x), float(y)]] * num_bboxes for y in range(grid_h) for x in range(grid_w)])
    y_true_class = y_true[..., 0:2]
    y_pred_class = y_pred[..., 0:2]   
    true_boxes = K.reshape(y_true[..., 3:], (-1, grid_w * grid_h, num_bboxes, 5))
    pred_boxes = K.reshape(y_pred[..., 3:], (-1, grid_w * grid_h, num_bboxes, 5))
    y_pred_xy = pred_boxes[..., 0:2] + K.variable(grid)
    y_pred_wh = pred_boxes[..., 2:4]
    y_pred_conf = pred_boxes[..., 4]
    y_true_xy = true_boxes[..., 0:2]
    y_true_wh = true_boxes[..., 2:4]
    y_true_conf = pred_boxes[..., 4]
    class_loss = K.sum(K.square(y_true_class - y_pred_class), axis=-1)
    xy_loss = K.sum(K.sum(K.square(y_true_xy - y_pred_xy), axis=-1) * y_true_conf, axis=-1)
    wh_loss = K.sum(K.sum(K.square(K.sqrt(y_true_wh) - K.sqrt(y_pred_wh)), axis=-1) * y_true_conf, axis=-1)
    inter_wh = K.maximum(K.zeros_like(y_pred_wh), (y_pred_wh + y_true_wh)/2 - K.square(y_pred_xy - y_true_xy))
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    true_area = y_true_wh[..., 0] * y_true_wh[..., 1]
    pred_area = y_pred_wh[..., 0] * y_pred_wh[..., 1]
    union_area = pred_area + true_area - inter_area
    iou = inter_area / union_area
    conf_loss = K.sum(K.square(y_true_conf * iou - y_pred_conf), axis=-1)
    loss = xy_loss + wh_loss + conf_loss + class_loss 
    return loss


def build_model(input_shape=(img_h, img_w, 3), grid_w=grid_w, grid_h=grid_h, num_bboxes=num_bboxes):
    i = Input(shape=input_shape)
    x = Conv2D(16, (1, 1))(i)
    x = Conv2D(32, (3, 3))(x)
    x = LeakyReLU(0.3)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3))(x)
    x = Conv2D(32, (3, 3))(x)
    x = LeakyReLU(0.3)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.1)(x) 
    x = Flatten()(x)
    x = Dense(256, activation='sigmoid')(x)
    x = Dense(grid_w * grid_h * (3 + num_bboxes * 5), activation='sigmoid')(x)
    x = Reshape((grid_w * grid_h, (3 + num_bboxes * 5)))(x)
    model = Model(i, x)
    adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss=yolo_loss, optimizer=adam)
    return model


if __name__=='__main__':
    x_train, y_train = build_dataset()
    img_dims = x_train[0].shape
    model = build_model()
    model.fit(x_train, y_train, batch_size=64, epochs=10)
    model.save_weights('yolo.h5')

