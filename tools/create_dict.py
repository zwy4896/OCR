#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   create_dict.py
@Time    :   2019/07/30 18:22:40
@Author  :   Zhang Wuyang
@Version :   1.0
'''

# here put the import lib

import os
import keys
import argparse
from config import Lang
from config import characters

parser = argparse.ArgumentParser()

parser.add_argument('--label_path', required=True, help='Ground truth path')
parser.add_argument('--key_path', required=True, help='OCR results')
arg = parser.parse_args()

# # path of labels.txt
# label_path = '/algdata02/algo-share/datasets/reader/testset/005ab/result/rec'
# # key.txt save path
# key_path = '/algdata02/wuyang.zhang/50.31/ocr_tf_crnn_ctc'

key = characters

dict_list = []
key_list = []
# for root, dirs, files in os.walk(label_path):
#     for file in files:
with open(os.path.join(arg.label_path, 'labels.txt'), 'r', encoding='utf-8') as f:
    while True:
        c = f.read(1)
        # print(c)
        if c != '$':
            dict_list.append(c)
        if not c:
            break
# print(dict_list)
for i in range(len(dict_list)):
    key_list.append(dict_list[i].strip())
# print(key_list)
str = set(key_list)
str_ = ''.join(str)

for char in str_:
    if char not in key:
        key += char
print(len(key))
# f = open('D:\Documents\\fanhan_data\\adjust\\zipin.txt', 'w', encoding='utf-8')
# f.write(str_)
i = 0
if os.path.exists(os.path.join(arg.key_path, Lang + '_keys.txt')):
    print(Lang + '_keys.txt already exists.')
    f = open(os.path.join(arg.key_path, Lang + '_keys.txt'), 'r', encoding='utf-8')
    lines = f.readline()
    f.close()
    # print(lines)
    with open(os.path.join(arg.key_path, Lang + '_keys.txt'), 'a+', encoding='utf-8') as nf:
        for char in str_:
            if char not in lines:
                print(char)
                nf.write(char)

elif not os.path.exists(os.path.join(arg.key_path, Lang + '_keys.txt')):
    f = open(os.path.join(arg.key_path, Lang + '_keys.txt'), 'w', encoding='utf-8')
    f.write(key)