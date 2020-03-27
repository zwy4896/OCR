data_dir="/algdata02/wuyang.zhang/_tfrecord/JPN" # tfrecord path
val_dir="/algdata02/wuyang.zhang/50.31/ocr_tf_crnn_ctc/Arabic_tfrecord" # validation sets path
model_dir="/algdata02/wuyang.zhang/50.31/tfrecord_bak/models-DENSENET_CTC-JPN-2020-03-16_17-45-19"  # If you want to restore, change this to your ckpt path
restore="False"   # Whether to restore
gpu_list="0,1,2,4"  # GPU list
batch_size=32   # batch size
lr=0.001    # learning rate
step_per_val=1000   # evaluation

python tools/train_densenetocr_ctc_multigpu.py \
--data_dir $data_dir \
--gpu_list "${gpu_list}[@]" \
--batch_size $batch_size \
--learning_rate $lr \
--step_per_val $step_per_val \
# --model_dir $model_dir \
# --restore $restore
