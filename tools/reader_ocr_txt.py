#coding=utf-8
import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
slim = tf.contrib.slim
sys.path.append(os.getcwd()+'/')

from nets.cnn.dense_net import DenseNet
from nets.cnn.resnet_v2 import ResNetV2
from nets.cnn.paper_cnn import PaperCNN
from nets.cnn.inception_v4 import InceptionV4

# import keys
import config as cfg


os.environ["CUDA_VISIBLE_DEVICES"]=""
# ------------------------------------Basic prameters------------------------------------
tf.app.flags.DEFINE_string(
    'image_dir', '', 'Path to the test sets directory .')
tf.app.flags.DEFINE_string(
    'detect_txt_dir', '', 'Path to the directory containing detection txt.')
tf.app.flags.DEFINE_string(
    'results_dir', '', 'Where to save the OCR results.')
tf.app.flags.DEFINE_string(
    'txt_dir', '', 'Where to save the OCR results in .')
# ------------------------------------Model infomation------------------------------------
tf.app.flags.DEFINE_string(
    'tflite_dir', '', 'Base directory for the tflite.')
tf.app.flags.DEFINE_string(
    'ckpt_dir', '', 'Base directory for the ckpt.')
tf.app.flags.DEFINE_boolean('debug', False, 'Debug mode')
tf.app.flags.DEFINE_string(
    'adj_dir', '', 'Base directory for txt line.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')

FLAGS = tf.app.flags.FLAGS
characters = cfg.characters
nclass = len(characters)
char_map_dict = {}
for i, val in enumerate(characters):
    char_map_dict[val] = i

def imageAdjust(img,boxes,is_book):
    newboxes=[]
    list_rotate=[]
    for box in boxes:
        if len(box)!=4:
            continue
        x,y,w,h = cv2.boundingRect(box)
        x=x-50
        y=y-50
        w=w+100
        h=h+100
        if x <0:
            x=0
        if y<0:
            y=0
        if w>img.shape[1]:
            w=img.shape[1]
        if h>img.shape[0]:
            h=img.shape[0]
        roi_bbox= img[y:y+h,x:x+w]
        rect = cv2.minAreaRect(box)
        newbox = np.int0(cv2.boxPoints(rect))
        angle = rect[2]
        if angle<-45:
            angle+=90
        center = (roi_bbox.shape[1] // 2, roi_bbox.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M = M.astype(np.float32)
        pt1 = np.array([box[0][0] - x, box[0][1] - y, 1]).astype(np.float32)
        pt2 = np.array([box[1][0] - x, box[1][1] - y, 1]).astype(np.float32)
        pt3 = np.array([box[2][0] - x, box[2][1] - y, 1]).astype(np.float32)
        pt4 = np.array([box[3][0] - x, box[3][1] - y, 1]).astype(np.float32)
        rotate_pt1 = np.matmul(M, pt1)
        rotate_pt2 = np.matmul(M, pt2)
        rotate_pt3 = np.matmul(M, pt3)
        rotate_pt4 = np.matmul(M, pt4)

        r_box = np.array([rotate_pt1,rotate_pt2,rotate_pt3,rotate_pt4]).astype(np.int32)

        r_x,r_y,r_w,r_h = cv2.boundingRect(r_box)
        # 右下扩边2像素
        r_w += 2
        r_h += 2

        if r_x <0:
            r_x=0
        if r_y<0:
            r_y=0
        if r_w>w:
            r_w=w
        if r_h>h:
            r_h=h
        rotate_bbox = cv2.warpAffine(roi_bbox, M, (w, h))
        roi_rotate= rotate_bbox[r_y:r_y+r_h,r_x:r_x+r_w]

        if roi_rotate.shape[0] > 1.5 * roi_rotate.shape[1]:
            roi_rotate = np.rot90(roi_rotate)
        newboxes.append(box)
        list_rotate.append(roi_rotate)
    return newboxes,list_rotate

def _int_to_string(value, char_map_dict=None):
    assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')
    
    return(characters[value])

def _sparse_matrix_to_list(sparse_matrix, char_map_dict=None):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    # the last index in sparse_matrix is ctc blanck note
    assert(isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')    

    dense_matrix =  len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]
    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(_int_to_string(val, char_map_dict))
        string_list.append(''.join(s for s in string if s != '*'))
    return string_list

def rec_process(modelDir, roi_rotate, rec_resizew, rec_resizeh):
    is_load_lite = False
    interpreter, input_details, output_details, is_load_lite = rec_load_tflite(modelDir)

    rat = rec_resizew / rec_resizeh
    if float(roi_rotate.shape[1]) / float(roi_rotate.shape[0]) > rat:
        dst = cv2.resize(roi_rotate, (rec_resizew, rec_resizeh), interpolation=cv2.INTER_AREA)
    else:
        x_height = float(rec_resizeh) / float(roi_rotate.shape[0])
        dst = cv2.resize(roi_rotate, (int(roi_rotate.shape[1] * x_height), rec_resizeh), interpolation=cv2.INTER_AREA)
        right = rec_resizew - dst.shape[1]
        dst = cv2.copyMakeBorder(dst, 0, 0, 0, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    X = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    X = X.reshape([rec_resizeh, rec_resizew, 1])
    X = np.expand_dims(X, axis=0)
    interpreter.set_tensor(input_details[0]['index'], X)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    if is_load_lite:
        y_pred = y_pred.reshape(1, y_pred.shape[0], nclass)
    out = rec_decode(y_pred)
    return out

def rec_ckpt_line(imgDir, resultFolder, ckpt):
    input_image = tf.placeholder(dtype=tf.float32, shape=[1, 32, None, 1])

    with tf.variable_scope(cfg.NET, reuse=False):
        if cfg.NET == 'DENSENET_CTC':
            net = DenseNet(input_image, is_training=False)
            cnn_out = net.net
        if cfg.NET == 'RESNET':
            net = ResNetV2(input_image, is_training = False)        
            cnn_out = net.net
        if cfg.NET == 'InceptionV4':
            net = InceptionV4(input_image, is_training = False)
            cnn_out = net.net
        if cfg.NET == 'CRNN':
            net = PaperCNN(input_image, is_training=False)
            cnn_out = net.net

        cnn_output_shape = tf.shape(cnn_out)
        batch_size = cnn_output_shape[0]
        cnn_output_h = cnn_output_shape[1]
        cnn_output_w = cnn_output_shape[2]
        cnn_output_channel = cnn_output_shape[3]

        # Get seq_len according to cnn output, so we don't need to input this as a placeholder
        seq_len = tf.ones([batch_size], tf.int32) * cnn_output_w

        # Reshape to the shape lstm needed. [batch_size, max_time, ..]
        cnn_out_transposed = tf.transpose(cnn_out, [0, 2, 1, 3])
        cnn_out_reshaped = tf.reshape(cnn_out_transposed, [batch_size, cnn_output_w, cnn_output_h * cnn_output_channel])

        cnn_shape = cnn_out.get_shape().as_list()
        cnn_out_reshaped.set_shape([None, cnn_shape[2], cnn_shape[1] * cnn_shape[3]])

        logits = slim.fully_connected(cnn_out_reshaped, nclass, activation_fn=None)

        # ctc require time major
        logits = tf.transpose(logits, (1, 0, 2))

    ctc_decoded, ctc_log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=True)

    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint(ckpt)

    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=save_path)
        write_txt=open(os.path.join(resultFolder, 'labels.txt'), 'w', encoding='UTF-8')

        for root, dirs, files in os.walk(imgDir):
            for path in files:
                if '.txt' in path:
                    print(path)
                    continue
                realPath = os.path.join(root, path)
                image_X = cv2.imread(realPath)
                try:
                    h, w, c = image_X.shape
                except AttributeError:
                    continue
                if h * w >= 32 * 32:
                    width = int(w * 32 / h)
                    image = cv2.resize(image_X, (width, 32))[:,:,::-1]
                    image = Image.fromarray(image).convert('L')
                    new_image = np.zeros((32, width))
                    new_image[:,0:width] = image
                    image = np.expand_dims(new_image, axis=0)
                    image = np.expand_dims(image, axis=-1)
                    image = np.array(image, dtype=np.float32)
                    if cfg.Lang == 'AB':
                        preds = sess.run(ctc_decoded[::-1], feed_dict={input_image:image})
                    else:
                        preds = sess.run(ctc_decoded, feed_dict={input_image:image})

                    preds = _sparse_matrix_to_list(preds[0], char_map_dict)
                    print('{}-->{}'.format(path, preds[0]))
                    if cfg.Lang == 'AB':
                        write_txt.write('{}$$${}\n'.format(path, preds[0][::-1].replace('#', '')))
                    else:
                        write_txt.write('{}$$${}\n'.format(path, preds[0].replace('#', '')))

def rec_load_tflite(graph):
    interpreter = tf.contrib.lite.Interpreter(graph)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    load = True
    return interpreter, input_details, output_details, load

def rec_decode(pred):
    char_list = []
    ori_char_list = []
    # print(type(pred))
    with tf.Session() as sess:
        pred_text = np.argmax(pred, axis=2)
    pred_text = pred_text[0]
    for i in range(len(pred_text)):
        ori_char_list.append(characters[pred_text[i]])
        if pred_text[i] != nclass-1 and (
                (not ((i > 0 and pred_text[i] == pred_text[i - 1])))):
            char_list.append(characters[pred_text[i]])
    if FLAGS.debug:
        print(pred_text)
        output = u''.join(ori_char_list)
    else:
        output = u''.join(char_list)

    return output

if __name__ == '__main__':
    rec_resizew = 768
    rec_resizeh = 32
    is_book = True
    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)
    if not os.path.exists(FLAGS.txt_dir):
        os.makedirs(FLAGS.txt_dir)
    if not os.path.exists(FLAGS.adj_dir):
        os.makedirs(FLAGS.adj_dir)        
    if cfg.MTHD == 'ckpt_line':
        rec_ckpt_line(FLAGS.image_dir, FLAGS.txt_dir, FLAGS.ckpt_dir)