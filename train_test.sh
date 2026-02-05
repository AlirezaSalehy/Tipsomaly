model_version='l14h' # l14h or l14

model_name=$1
log_dir=$2

device=$3
epoch=$4

# Train
python train.py --model_name $model_name --dataset mvtec --device $device --cls_seg_los seg --l1_lambda 0.0 --d_deep_tokens 0 --n_deep_tokens 0 --epoch 5 --model_version $model_version --fixed_prompt_type industrial
python train.py --model_name $model_name --dataset visa --device $device --cls_seg_los seg --l1_lambda 0.0 --d_deep_tokens 0 --n_deep_tokens 0 --epoch 5 --model_version $model_version --fixed_prompt_type industrial

# Test
python test.py --checkpoint_path ./workspaces/trained_on_visa_$model_name/$log_dir/checkpoints --dataset mvtec --devices $device --epoch $epoch --model_version $model_version --fixed_prompt_type industrial
for dataset in visa mpdd btad sdd dagm dtd; do
    python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints --dataset $dataset --devices $device  --epoch $epoch --model_version $model_version --fixed_prompt_type industrial
done

for dataset in isic tn3k cvc-colondb cvc-clinicdb; do
    python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints  --dataset $dataset --devices $device --fixed_prompt_type industrial --epoch $epoch --model_version $model_version
done

# Using medical prompts
for dataset in headct brainmri br35h; do
    python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints  --dataset $dataset --devices $device --fixed_prompt_type medical --epoch $epoch --model_version $model_version
done
