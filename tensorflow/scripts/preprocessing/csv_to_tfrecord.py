import tensorflow as tf
import numpy as np
import base64
import csv
import os
from PIL import Image
import io

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class_dict = {
    1: b'squirrel', # List of class map Text with byte
}

def create_tf_example(img_name, img_path, labels):
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    image_format = str.encode(img_name.split('.')[1])

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for label in labels:
        xmins.append(float(int(label[0][4]) / int(label[0][1])))
        xmaxs.append(float(int(label[0][6]) / int(label[0][1])))
        ymins.append(float(int(label[0][5]) / int(label[0][2])))
        ymaxs.append(float(int(label[0][7]) / int(label[0][2])))
        
        classes_text.append(label[0][3])

        class_ = 0
        if (label[0][3] == "squirrel"):
            class_ = 1

        classes.append(class_)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(int(labels[0][0][2])),
        'image/width': int64_feature(int(labels[0][0][1])),
        'image/filename': bytes_feature(str.encode(labels[0][0][0])),
        'image/source_id': bytes_feature(labels[0][0][0]),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    
    return tf_example

def checkHowManyAhead(csv_reader, i, relative):
    try:
        if csv_reader[i + relative][0][0] == csv_reader[i + relative + 1][0][0]:
            return checkHowManyAhead(csv_reader, i, relative + 1)
        else:
            return relative
    except IndexError:
        print "End of list"
        return 0

def main():
    writer = tf.python_io.TFRecordWriter('train.tfrecords')

    data_path = '../../workspace/squirrel_detector/images/Squirrel'
    images = os.listdir(data_path)

    label_csv = '../../workspace/squirrel_detector/annotations/train_labels.csv'
    csvee_reader = csv.reader(open(label_csv, 'r'))

    csv_reader = zip(csvee_reader)

    for i in range(0, len(images)):
        labels = []
        ahead = checkHowManyAhead(csv_reader, i, 0)
        for j in range(0, ahead + 1):
            labels.append(csv_reader[i + j])
        tf_example = create_tf_example(images[i], data_path + "/" + images[i], labels)
        writer.write(tf_example.SerializeToString())

    writer.close()

main()
