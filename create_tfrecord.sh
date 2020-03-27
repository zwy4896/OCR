# Train data directories
dir=(
    "training_data"
)
# Data augmentation
method=(
    # "MotionBlur"
    # "ColorTemp"
    # "HUE"
)
# tfrecord save path
data_dir=""
max_label_lenth=60
min_label_length=6
for i in "${dir[@]}"
do
    echo $i
    image_dir="/PATH_TO_YOUR_IMG/$i"
    anno_file="/PATH_TO_YOUR_IMG/$i/labels.txt"
    echo $image_dir
    echo $anno_file
    # create original tfrecord
    python tools/create_tfrecord.py --image_dir $image_dir --anno_file $anno_file \
    --data_dir $data_dir --max_label_lenth $max_label_lenth --min_label_length $min_label_length
    if [ $? -eq 0 ]; then
        echo "succeed"
    else
        echo "failed"
        break
    fi
    # for j in "${method[@]}"
    # do
    #     echo $j
    #     # data augmentation
    #     python tools/create_crnn_ctc_tfrecord.py --image_dir $image_dir --anno_file $anno_file \
    #     --data_dir $data_dir --aug True --$j True
    # done
done