data_dir="" # tfrecord path
val_dir="" # validation sets path
model_dir=""  # If you want to restore, change this to your ckpt path. Otherwise, just let it go!
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
