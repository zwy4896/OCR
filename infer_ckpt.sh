date=$(date "+%m%d")
image_dir=""
results_dir=""
txt_dir=""
adj_dir=""
ckpt_dir=""

gt=""
dt=$txt_dir
save_path=$results_dir
mod="acc"
# mod="lev"
# Debug mode --debug is OKay
debug="True"

python tools/reader_ocr_txt.py \
--image_dir $image_dir \
--detect_txt_dir $detect_txt_dir \
--results_dir $results_dir \
--txt_dir $txt_dir \
--ckpt_dir $ckpt_dir \
--adj_dir $adj_dir
echo Inference done!

python tools/accuracy_test.py --gt $gt --dt $dt --s $save_path --mod $mod