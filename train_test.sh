model_name="prompt_learning_segonly"
log_dir="cup-mountain"

# model_name="prompt_learning"
# # log_dir="winter-purple" # "michigan-wisconsin" -> the cls loss was detaches!!

# model_name="prompt_learning_clsonly"
# log_dir="kitten-dakota" 
device=2
epoch=4

# python train.py --model_name $model_name --dataset mvtec --device $device --cls_seg_los both
# python train.py --model_name $model_name --dataset visa --device $device --cls_seg_los both

python test.py --checkpoint_path ./workspaces/trained_on_visa_$model_name/$log_dir/checkpoints --dataset mvtec --devices $device --epoch $epoch
for dataset in visa mpdd btad sdd dagm dtd; do
    python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints --dataset $dataset --devices $device  --epoch $epoch
done