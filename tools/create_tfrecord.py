"""
Write text features and labels into tensorflow records
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import json

import tensorflow as tf

import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

import keys
import config as cfg
import re
import time
# import keys_de as keys
#sys.path.append(os.getcwd()+'/tools')

_IMAGE_HEIGHT = 32

tf.app.flags.DEFINE_string(
    'image_dir', '/algdata02/algo-share/datasets/reader/trainset/0018ab/result/rec', 'Dataset root folder with images.')
tf.app.flags.DEFINE_string(
    'anno_file', '/algdata02/algo-share/datasets/reader/trainset/0018ab/result/rec/labels.txt', 'Path of dataset annotation file.')
tf.app.flags.DEFINE_string(
    'data_dir', '/algdata02/wuyang.zhang/50.31/ocr_tf_crnn_ctc/Arabic_tfrecord', 'Directory where tfrecords are written to.')
tf.app.flags.DEFINE_float(
    'validation_split_fraction', 0, 'Fraction of training data to use for validation.')
tf.app.flags.DEFINE_float(
    'max_label_lenth', 60, 'Maximum length of label')
tf.app.flags.DEFINE_float(
    'min_label_lenth', 3, 'Maximum length of label')
tf.app.flags.DEFINE_boolean(
    'shuffle_list', True, 'Whether shuffle data in annotation file list.')
tf.app.flags.DEFINE_boolean(
    'aug', False, 'Data augmentation. ')
tf.app.flags.DEFINE_boolean(
    'MotionBlur', False, 'Data augmentation. ')
tf.app.flags.DEFINE_boolean(
    'ColorTemp', False, 'Data augmentation. ')
tf.app.flags.DEFINE_boolean(
    'HUE', False, 'Data augmentation. ')
  
tf.app.flags.DEFINE_string(
    'char_map_json_file', './char_map/char_map.json', 'Path to char map json file')

FLAGS = tf.app.flags.FLAGS

characters = cfg.characters
# print(characters)
nclass = len(characters)
# print(nclass)
char_map_dict = {}
for i, val in enumerate(characters):
    char_map_dict[val] = i

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _string_to_int(label):
    # convert string label to int list by char map
    #char_map_dict = json.load(open(FLAGS.char_map_json_file, 'r'))
    int_list = []
    for c in label:
        int_list.append(char_map_dict[c])
    return int_list

# 解析textrender的格式, 文件名(不含后缀) 文本内容
def parse_textrender_label(line):
    line = line.strip()
    try:
        image_name = line.split('$$$')[0]
        label = line.split('$$$')[1].strip()
        label = re.sub('[\u2002\t\u2009]', ' ', label)
        label = re.sub('[\u3000\xa0]', '', label)
    except IndexError:
        print(image_name)
    #     image_name = line.split('$$')[0]
    #     label = line.split('$$')[1].strip()
    #     label = re.sub('[\u3000\xa0]', '', label)
    # label = line.split(' ')[1].lower()

    return image_name, label

def _write_tfrecord(dataset_split, anno_lines):
    labels_list = []
    width_list = []
    long_label = []
    long_pic = []

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    if  not FLAGS.aug:
        fix = '.tfrecord'
    else:
        if FLAGS.MotionBlur:
            fix = '_motion_blur.tfrecord'
        if FLAGS.ColorTemp:
            fix = '_color_temp.tfrecord'
        if FLAGS.HUE:
            fix = '_hue.tfrecord'
    tfrecords_path = os.path.join(FLAGS.data_dir, dataset_split + '-' + cfg.Lang + '-' 
                    + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())) + fix)
    # 记录tfrecord信息，图片最大宽度
    tfrecord_info = open(os.path.join(FLAGS.data_dir, 'info.txt'), 'a+', encoding='utf8')
    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for i, line in enumerate(anno_lines):
            line = line.strip()
            image_name, label = parse_textrender_label(line)
            if label == '卍':
                continue
            # 如果label中含有模糊行，整行过滤
            if '###' in label:
                continue
            ########## !!!!暂时处理，需要删去 #############
            if 'ô' in label:
                continue
            ##########################################
            # 阿拉伯语label要特殊处理
            if cfg.Lang == 'AB':
                label = label[::-1]
            # label最大字符长度，超过此长度过滤
            if len(label) > FLAGS.max_label_lenth:
                continue
            # label最短字符长度，小于此长度过滤
            if len(label) < FLAGS.min_label_lenth:
                continue
            image_path = os.path.join(FLAGS.image_dir, image_name)

            image = cv2.imread(image_path)
            if image is None: 
                continue # skip bad image.

            h, w, c = image.shape
            if w > h:
                height = _IMAGE_HEIGHT
                width = int(w * height / h)
                labels_list.append(len(label))
                width_list.append(width)
                image = cv2.resize(image, (width, height))
                if not FLAGS.aug:
                    is_success, image_buffer = cv2.imencode('.jpg', image)
                else:
                    if FLAGS.MotionBlur:
                        aug = iaa.MotionBlur(k = 10, angle = np.random.randint(-45, 45))
                    if FLAGS.ColorTemp:
                        aug = iaa.ChangeColorTemperature((1100, 10000))
                    if FLAGS.HUE:
                        aug = iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
                    img_aug = aug(image = image)
                    is_success, image_buffer = cv2.imencode('.jpg', img_aug)
                if not is_success:
                    continue
                # convert string object to bytes in py3
                image_name = image_name if sys.version_info[0] < 3 else image_name.encode('utf-8') 
                features = tf.train.Features(feature={
                   'labels': _int64_feature(_string_to_int(label)),
                   'images': _bytes_feature(image_buffer.tostring()),
                   'imagenames': _bytes_feature(image_name)
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                sys.stdout.write('\r>>Writing to {:s}.tfrecords {:d}/{:d}'.format(dataset_split, i + 1, len(anno_lines)))
                sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.write('>> {:s}.tfrecords write finish.\n'.format(dataset_split))
        sys.stdout.flush()
        if width_list:
            tfrecord_info.write('{}: {}\n'.format(tfrecords_path.split('/')[-1], max(width_list)))

def _convert_dataset():
    with open(FLAGS.anno_file, 'r', encoding='utf8') as anno_fp:
        anno_lines = anno_fp.readlines()    

    if FLAGS.shuffle_list:
        random.shuffle(anno_lines)
    
    # split data in annotation list to train and val
    split_index = int(len(anno_lines) * (1 - FLAGS.validation_split_fraction))
    train_anno_lines = anno_lines[:split_index - 1]
    validation_anno_lines = anno_lines[split_index:]

    dataset_anno_lines = {'train' : train_anno_lines, 'validation' : validation_anno_lines}
    for dataset_split in ['train']:
        _write_tfrecord(dataset_split, dataset_anno_lines[dataset_split])

def main(unused_argv):
    _convert_dataset()

if __name__ == '__main__':
    tf.app.run()
