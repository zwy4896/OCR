# A easy-to-use multi-network OCR training script

You can train a deep-learning OCR model using only three bash files.
Surpported network: DenseNet, crnn, InceptionV4, mobileNetV3

## 1. Create keys

    Using create_dict.sh to obtain keys
`. create_dict.sh`

## 2. Create TFRecord
What is [TFRecord](https://tensorflow.google.cn/tutorials/load_data/tfrecord)?
    
    Using create_tfrecord.sh to convert original training data to tfrecord
`. create_tfrecord.sh`

## 3. Edit *tools/config.py*

## 4. Enjoy it!
 `. train.sh`