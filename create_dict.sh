# Train data directories
dir=(
    "training_data"
)

key_path="PATH/TO/SAMPLE/"
# tfrecord save path
for i in "${dir[@]}"
do
    echo Create dict
    label_path="PATH/TO/SAMPLE/$i"
    python tools/create_dict.py --label_path $label_path --key_path $key_path
done