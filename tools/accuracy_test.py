# coding:utf-8
# encoding:utf-8
# -*- coding: utf-8 -*-
"""
Created on MON JUN 10 2019

@author: Wuyang.Zhang
"""

import os
import numpy as np
import re
from shapely.geometry import Polygon, MultiPoint
import shapely
import keys
import math
import shutil
import config as cfg
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gt', required=True, help='Ground truth path')
parser.add_argument('--dt', required=True, help='OCR results')
parser.add_argument('--s', required=True, help='Save path')
parser.add_argument('--mod', required=True, help='Save path')
arg = parser.parse_args()

def get_intersection_over_union(pD, pG):
    a = np.array(pG).reshape(4, 2)  # 四边形二维坐标表示
    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上

    b = np.array(pD).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    union_poly = np.concatenate((a, b))   #合并两个box坐标，变为8*2
    #print(union_poly)
    #print(MultiPoint(union_poly).convex_hull)      #包含两四边形最小的多边形点
    # poly1 = polygon_from_points(pD)
    # poly2 = polygon_from_points(pG)
    if not poly1.intersects(poly2): #如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area   #相交面积
            # print(inter_area)
            union_area = poly1.area + poly2.area - inter_area
            # union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou= 0
            iou=float(inter_area) / union_area
            # iou=float(inter_area) /(pD.area+pG.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def ParseResult(filePath, dstPath, flag):
    vec_txt = []
    vec_pos = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if 'E/目录:: ' in line:
                txt_name = line.split(':: ')[1].split('.')[0]
                flag = False
            elif line.find('line points') != -1:
                textStr = line.split(': ')[2]
                vec_pos.append(textStr)
                flag = True
            elif line.find('    >>>	Line text') != -1:
                textStr = line.split(']: ')[1]
                vec_txt.append(textStr)
                flag = True
            # else:
            #     flag = True

            if flag:
                # print(txt_name)
                with open(dstPath + txt_name + '.txt', 'a+', encoding='utf-8') as f:
                    if len(vec_pos):
                        for pos in vec_pos:
                            f.write('pos: ' + pos)
                    if len(vec_txt):
                        for txt in vec_txt:
                            f.write('txt: ' + txt)

                vec_txt = []
                vec_pos = []

                flag = False

def ReadLine(dstFile, gtFile):
    # Get line in dst file
    dst_labels = []
    gt_labels = []

    for file_name in os.listdir(gtFile):
        with open(dstFile + file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                dst_labels.append(line)

        with open(gtFile + file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                gt_labels.append(line)
        # count += 1

    # print(dst_labels)

        print(file_name)
        LevDistance(dst_labels, gt_labels)
        dst_labels = []
        gt_labels = []

# 编辑距离
def LevDistance(gtLine, dtLine):
    '''
    :param dstLine: ocr识别文本
    :param gtLine:  gt文本
    :return:        单行accuracy
    '''
    dp = np.array(np.arange(len(gtLine)+1))
    for i in range(1, len(dtLine)+1):
        temp1 = dp[0]
        dp[0] += 1
        for j in range(1, len(gtLine)+1):
            temp2 = dp[j]
            if dtLine[i-1] == gtLine[j-1]:
                dp[j] = temp1
            else:
                dp[j] = min(temp1, min(dp[j-1], dp[j]))+1
            temp1 = temp2

   # if 1 - dp[len(dstLine)] / len(dstLine) > 0.8:

    accuracy = 1 - dp[len(gtLine)] / len(dtLine)
    if math.isnan(accuracy):
        accuracy = 1
    if accuracy < 0:
        accuracy = 0

    return accuracy

def AccuracyCompute(gtxt, dtxt):
    '''
    accuracy
    :param gtxt: ground truth文本
    :param dtxt: 识别文本
    :return: 单行accuracy
    '''

    gt_dict = {}     # 字典，存放字频
    dt_dict = {}

    # correct_count = 0
    err_count = 0
    total_count = 0
    count = len(gtxt)

    total_count += count

    for key in char:
        # print(key)
        # 初始化dt gt字典，5990
        gt_dict[key] = 0
        dt_dict[key] = 0

    for gt_key in gtxt:
        # print(key)
        if gt_key in gt_dict:
            gt_dict[gt_key] += 1
        else:
            gt_dict.update(gt_key = 1)
            dt_dict.update(gt_key = 1)
    for dt_key in dtxt:
        if dt_key in dt_dict:
            dt_dict[dt_key] += 1
        else:
            gt_dict.update(dt_key = 1)
            dt_dict.update(dt_key = 1)

    gt_cache = [key for key, key in gt_dict.items()]
    dt_cache = [key for key, key in dt_dict.items()]

    for i in range(len(gt_cache)):
        if gt_cache[i] - dt_cache[i]:
            err_count += abs(gt_cache[i] - dt_cache[i])

    # print(sum(values))
    if len(gtxt):
        accuracy = 1 - err_count / count
    else:
        accuracy = 0
    if accuracy < 0:
        accuracy = 0

    return accuracy

if __name__ == '__main__':
    # char = keys.alphabet_AB[:]  # 5990字典
    symbol = ["_,$%^*\"\'—！:：【】[]|〔「」［］«‹»《》–><‘'’；;“”，·。.？?()、-~@：❋￥%……&*（）]・"]
    char = cfg.characters
    high = []
    if not os.path.exists(arg.s):
        os.makedirs(arg.s)
    gt_name = os.listdir(arg.gt)
    dt_name = os.listdir(arg.dt)
    thres = 0.5    # IOU阈值
    acc_list = []
    total_accuarcy_list = {}
    # t1 = '看起来会好像我们增'
    # t2 = '看来会好像我们'
    # acc = LevDistance(t1, t2)
    # print(acc)
    dt_list = []
    gt_list = []
    for name in gt_name:
        nf = open(os.path.join(arg.s, name), 'w', encoding='utf-8')
        print(name)
        total_accuarcy = []
        gt_dict = []
        with open(os.path.join(arg.gt, name), 'r', encoding='utf-8') as f:
            for gt_l in f:
                if not ('book' in gt_l or 'line' in gt_l):
                    gt_dict.append(gt_l.split('$$$')[0])
                    gt_dict.append(gt_l.split('$$$')[1])
            gt_coor = gt_dict[::2]
            gt_txt = gt_dict[1::2]
            # print(gt_txt)
        dt_dict = []
        with open(os.path.join(arg.dt, name), 'r', encoding='utf-8') as f:
            for dt_l in f:
                if '卍' in dt_l:
                    dt_dict.append(dt_l.split('卍')[0])
                    dt_dict.append(dt_l.split('卍')[1])
                else:
                    dt_dict.append(dt_l.split('$$$')[0])
                    dt_dict.append(dt_l.split('$$$')[1])
            dt_coor = dt_dict[::2]
            dt_txt = dt_dict[1::2]
            # print(dt_txt)
        # IOU匹配ground truth
        if cfg.MTHD == 'tflite' or cfg.MTHD == 'ckpt':
            for i in range(len(gt_coor)):
                is_continue = True
                for j in range(len(dt_coor)):
                    iou = get_intersection_over_union(gt_coor[i].split(','), dt_coor[j].split(','))    # 计算IOU
                    # print(iou)
                    if iou > thres:
                        gt_t = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！:：\【\】\[\]|〔「」［］«‹»《》☆ー※–><‘'’；;“”，·。.？?()、\-~@：❁❋￥%……&*（）・]+", "", gt_txt[i])
                        dt_t = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！:：\【\】\[\]|〔「」［］«‹»《》☆ー※–><‘'’；;“”，·。#＃.？?()、\-~@：❁❋￥%……&*（）・]+", "", dt_txt[j])
                        gt_t = gt_t.lower()
                        dt_t = dt_t.lower()
                        dt_list.append(dt_t)
                        gt_list.append(gt_t)
                        if is_continue and '#' not in gt_t and '＃' not in gt_t:
                            print('gt: ', gt_t, 'dt: ', dt_t)
                            if arg.mod == 'acc':
                                acc_line = AccuracyCompute(gt_t, dt_t)    # 单行accuracy
                            if arg.mod == 'lev':
                                acc_line = LevDistance(gt_t, dt_t)
                            nf.write('gt: ' + gt_t + ' ' + 'dt: ' + dt_t + '\n')
                            nf.write('line accuracy: ' + str(acc_line) + '\n')
                            print('准确率: ', acc_line)
                            total_accuarcy.append(acc_line)
                        else:
                            continue
        if cfg.MTHD == 'tflite_line' or cfg.MTHD == 'ckpt_line':
            # Matching label by text line name
            for i in range(len(gt_coor)):
                for j in range(len(dt_coor)):
                    if gt_coor[i] == dt_coor[j]:
                        is_continue = True
                        gt_t = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！:：\【\】\[\]|〔「」［］«‹»《》☆ー※–><‘'’；❋;“”，·。.？?()、\-~@：❁❉￥%……&*（）・]+", "", gt_txt[i])
                        dt_t = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！:：\【\】\[\]|〔「」［］«‹»《》☆ー※–><‘'’；❋#;“”，·。.？?#()、\-~@：❁❉￥%……&*（）・]+", "", dt_txt[j])
                        gt_t = gt_t.lower()
                        dt_t = dt_t.lower()
                        dt_list.append(dt_t)
                        gt_list.append(gt_t)

                        if is_continue and '#' not in gt_t and '＃' not in gt_t:
                            print('gt: ', gt_t, 'dt: ', dt_t)
                            if arg.mod == 'acc':
                                acc_line = AccuracyCompute(gt_t, dt_t)    # 单行accuracy
                            if arg.mod == 'lev':
                                acc_line = LevDistance(gt_t, dt_t)
                            # if acc_line > 0.8:
                            nf.write(gt_coor[i] + '----------\n')
                            nf.write('gt: ' + gt_t + ' ' + 'dt: ' + dt_t + '\n')
                            nf.write('line accuracy: ' + str(acc_line) + '\n')
                            nf.write('-----------------------------\n')
                            print('准确率: ', acc_line)
                            total_accuarcy.append(acc_line)
                        else:
                            continue
        acc_sum = 0
        for i in total_accuarcy:
            # print(i)
            acc_sum += i
        if len(total_accuarcy):
            acc_write = acc_sum / len(total_accuarcy)
            acc_list.append(acc_write)
        else:
            acc_write = 0
            acc_list.append(acc_write)
        # print('acc_list: ', acc_list)
        acc_avg = sum(acc_list) / len(acc_list)
        total_accuarcy_list[name] = acc_write
        print('综合accuracy: ', acc_avg)    # 计算测试集总体准确率
    with open(os.path.join(arg.s, '%.4f'%acc_avg + '_results.txt'), 'w', encoding='utf-8') as f:
        for name in total_accuarcy_list:
            f.write(name + '\n' + 'Total accuracy: ' + ' ' + str(total_accuarcy_list[name]) + '\n\n')