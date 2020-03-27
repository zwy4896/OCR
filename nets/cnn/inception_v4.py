#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
import time

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers import xavier_initializer
from collections import namedtuple
from tensorflow.contrib.layers.python.layers import utils

relu = tf.nn.relu
IncepParams = namedtuple('DensenetParameters', ['dropout_keep_prob',
                                                'weight_decay',
                                                'final_endpoint',
                                                   ])

def block_inception_a(inputs, scope=None):
    """Builds Inception-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInceptionA', [inputs]):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 96, [1, 1], scope='Conv2d_0b_1x1')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

def block_reduction_a(inputs, stride=2, scope=None):
    """Builds Reduction-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockReductionA', [inputs]):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 384, [3, 3], stride=stride, padding='SAME',
                                     scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                branch_1 = slim.conv2d(branch_1, 256, [3, 3], stride=stride,
                                     padding='SAME', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(inputs, [3, 3], stride=stride, padding='SAME',
                                         scope='MaxPool_1a_3x3')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

def block_inception_b(inputs, scope=None):
    """Builds Inception-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInceptionB', [inputs]):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 224, [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 256, [7, 1], scope='Conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, 224, [1, 7], scope='Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, 224, [7, 1], scope='Conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, 256, [1, 7], scope='Conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
        return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

def block_reduction_b(inputs, stride=2, scope=None):
    """Builds Reduction-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockReductionB', [inputs]):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                branch_0 = slim.conv2d(branch_0, 192, [3, 3], stride=stride,
                               padding='SAME', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 256, [1, 7], scope='Conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 320, [7, 1], scope='Conv2d_0c_7x1')
                branch_1 = slim.conv2d(branch_1, 320, [3, 3], stride=stride,
                               padding='SAME', scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.max_pool2d(inputs, [3, 3], stride=stride, padding='SAME',
                                   scope='MaxPool_1a_3x3')
        return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

def block_inception_c(inputs, scope=None):
    """Builds Inception-C block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                        stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInceptionC', [inputs]):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = tf.concat(axis=3, values=[
                  slim.conv2d(branch_1, 256, [1, 3], scope='Conv2d_0b_1x3'),
                  slim.conv2d(branch_1, 256, [3, 1], scope='Conv2d_0c_3x1')])
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 448, [3, 1], scope='Conv2d_0b_3x1')
                branch_2 = slim.conv2d(branch_2, 512, [1, 3], scope='Conv2d_0c_1x3')
                branch_2 = tf.concat(axis=3, values=[
                  slim.conv2d(branch_2, 256, [1, 3], scope='Conv2d_0d_1x3'),
                  slim.conv2d(branch_2, 256, [3, 1], scope='Conv2d_0e_3x1')])
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 256, [1, 1], scope='Conv2d_0b_1x1')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

def conv(x, channels, kernel_size, stride, scope, normalizer_fn=slim.batch_norm, activation_fn=relu):
    return slim.conv2d(x, channels, kernel_size, stride, scope=scope, normalizer_fn=normalizer_fn, activation_fn=activation_fn)

class InceptionV4(object):
    default_params = IncepParams(
        dropout_keep_prob=0.5,
        weight_decay=0.00004,
        final_endpoint='Mixed_7d'
    )
    def __init__(self, inputs, is_training):
        self._scope = 'InceptionV4'
        self.build_net(inputs, is_training)
        self.params = InceptionV4.default_params

    def build_net(self, inputs, is_training):
        """
        Core default tensorflow model for text recognition.
        Args:
            images: input images
            labels: input groundtruth labels 
            wemb_size: word embedding size
            seq_len: max sequence length for lstm with end of sequence
            num_classes: text label classes
            lstm_size: lstm size
            is_training: tensorflow placeholder
            dropout_keep_prob: tensorflow placeholder for dropout
            weight_decay: tensorflow model weight decay factor
            final_endpoint: final endpoint for CNN(InceptionV4)
            name: name scope
            reuse: reuse parameter
        Returns:
            output_array: (batch, seq_len, num_classes) logits
            attention_array: (batch, h, w, seq_len) attention feature map
        """
        end_points = {}

        def add_and_check_final(name, net):
          end_points[name] = net
          return name == self.params.final_endpoint

        with tf.variable_scope(self._scope, 'InceptionV4', [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                outputs_collections=end_points_collection, stride=1, padding='SAME'):
                # 299 x 299 x 3
                # 32 x 768 x 1
                net = slim.conv2d(inputs, 32, [3, 3], stride=2,
                                  padding='SAME', scope='Conv2d_1a_3x3')
                # print(net.shape)
                # 149 x 149 x 32
                # 28 x 764 x 32
                net = slim.conv2d(net, 32, [3, 3], padding='SAME',
                                  scope='Conv2d_2a_3x3')
                # print(net.shape)
                # 147 x 147 x 32
                # 28 x 764 x 64
                net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_2b_3x3')
                # print(net.shape)

                # 147 x 147 x 64
                with tf.variable_scope('Mixed_3a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME',
                                               scope='MaxPool_0a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 96, [3, 3], stride=2, padding='SAME',
                                           scope='Conv2d_0a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                    # print(net.shape)

                # 73 x 73 x 160
                with tf.variable_scope('Mixed_4a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='SAME',
                                             scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], padding='SAME',
                                             scope='Conv2d_1a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                    # print(net.shape)

                # 71 x 71 x 192
                with tf.variable_scope('Mixed_5a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [3, 3], stride=2, padding='SAME',
                                           scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME',
                                               scope='MaxPool_1a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                    # print(net.shape)

                # 35 x 35 x 384
                # 4 x Inception-A blocks
                for idx in range(4):
                    block_scope = 'Mixed_5' + chr(ord('b') + idx)
                    net = block_inception_a(net, block_scope)
                    # print(net.shape)

                # 35 x 35 x 384
                # Reduction-A block
                net = block_reduction_a(net, 1, 'Mixed_6a')
                # print(net.shape)

                # 17 x 17 x 1024
                # 7 x Inception-B blocks
                for idx in range(7):
                    block_scope = 'Mixed_6' + chr(ord('b') + idx)
                    net = block_inception_b(net, block_scope)
                    # print(net.shape)

                # 17 x 17 x 1024
                # Reduction-B block
                net = block_reduction_b(net, 1, 'Mixed_7a')
                # print(net.shape)

                # 8 x 8 x 1536
                # 3 x Inception-C blocks
                for idx in range(3):
                    block_scope = 'Mixed_7' + chr(ord('b') + idx)
                    net = block_inception_c(net, block_scope)
                    # print(net.shape)
                self.end_points = utils.convert_collection_to_dict(end_points_collection)
                self.net = net

     
if __name__=='__main__':
    os.environ['CUDA_VISIBILE_DEVICES']='0'
    
    inputs = tf.placeholder(tf.float32, [1, 32, 768, 3], name='inputs')
    is_training = tf.placeholder(tf.bool, name="is_training")
    xception = InceptionV4(inputs, is_training)
    print(xception)