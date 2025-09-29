# model_name="prompt_learning_segonly_518_l1norm003"
# log_dir="enemy-orange"

# model_name="prompt_learning_segonly_518_l1norm01"
# log_dir="robert-oscar"

# model_name="prompt_learning_segonly_518_l1norm001"
# log_dir="harry-connecticut"

# model_name="prompt_learning_segonly_518"
# log_dir="vegan-arkansas"

# model_name="prompt_learning_segonly"
# log_dir="cup-mountain"
# model_name="prompt_learning"
# # log_dir="winter-purple" # "michigan-wisconsin" -> the cls loss was detaches!!

# model_name="prompt_learning_clsonly"
# log_dir="kitten-dakota" 

# model="prompt_learning_deeptuning_segonly_518"
# log_fir="three_mockingbird"

model_name=$1
log_dir=$2

device=$3
epoch=$4

# python train.py --model_name $model_name --dataset mvtec --device $device --cls_seg_los seg --l1_lambda 0.0 --d_deep_tokens 0 --n_deep_tokens 0 --epoch 5 --model_version 'l14h'
# python train.py --model_name $model_name --dataset visa --device $device --cls_seg_los seg --l1_lambda 0.0 --d_deep_tokens 0 --n_deep_tokens 0 --epoch 5 --model_version 'l14h'
# python test.py --checkpoint_path ./workspaces/trained_on_visa_$model_name/$log_dir/checkpoints --dataset mvtec --devices $device --epoch $epoch --model_version 'l14h'
# for dataset in visa; do
#     python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints --dataset $dataset --devices $device  --epoch $epoch --model_version 'l14h'
# done

# ADD VISA AGAIN
python test.py --checkpoint_path ./workspaces/trained_on_visa_$model_name/$log_dir/checkpoints --dataset mvtec --devices $device --epoch $epoch --model_version 'l14h'
for dataset in visa mpdd btad sdd dagm dtd; do
    python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints --dataset $dataset --devices $device  --epoch $epoch --model_version 'l14h'
done

for dataset in brainmri headct br35h isic tn3k cvc-colondb cvc-clinicdb; do
    python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints --dataset_category med --dataset $dataset --devices $device  --epoch $epoch --model_version 'l14h'
done
# # closeness prevention regulation 
