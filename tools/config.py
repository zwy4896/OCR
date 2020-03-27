#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2020/01/20 09:48:40
@Author  :   Zhang Wuyang
@Version :   1.0
'''

# here put the import lib

import keys
# Language
Lang = 'CN'

characters = keys.alphabet_blank[:] + u'卍'

# Network architecture
'''
Copy follwing network into NetWork: 
    DENSENET_CTC
    RESNET
    InceptionV4
    CRNN(Without RNN)
'''
NetWork = 'DENSENET_CTC'

# Max width & max label length (int)
# Run create_crnn_ctc_tfrecord.py to obtain the value
MaxWid = 3113
MaxLen = 60

# Logger config
logPath = ''
logTaskName = NetWork + '-' + Lang
logTrainMemberName = ''