# This script provides the command to train your model

# By setting "model_name" and "log_dir" you define the path for the checkpoints to be saved and 
# you can use the same values later to test on other datasets in a loop easily  
model_version='l14h' # l14h or l14
model_name=$1
epoch=$2

models_dir="/kaggle/working/tips"
data_root_dir="/kaggle/working/datasets"
# checkpoint_path="/kaggle/working/checkpoints" # uncomment for testing results

# Train on MVTec
python train.py --models_dir $models_dir --model_name $model_name --data_root_dir $data_root_dir --dataset mvtec --cls_seg_los seg --l1_lambda 0.0 --d_deep_tokens 0 --n_deep_tokens 0 --epoch $epoch --model_version $model_version --fixed_prompt_type industrial

# Train on VisA
# python train.py --models_dir $models_dir --model_name $model_name --data_root_dir $data_root_dir --dataset visa --cls_seg_los seg --l1_lambda 0.0 --d_deep_tokens 0 --n_deep_tokens 0 --epoch $epoch --model_version $model_version --fixed_prompt_type industrial


# Test multiple datasets in a loop

# for dataset in visa mpdd btad sdd dagm dtd; do
#     python test.py --models_dir $models_dir --checkpoint_path $checkpoint_path --data_root_dir $data_root_dir --dataset $dataset --epoch $epoch --model_version $model_version --fixed_prompt_type industrial
# done

# Medical segmentation datasets - using learned prompts
# for dataset in isic tn3k cvc-colondb cvc-clinicdb; do
#     python test.py --models_dir $models_dir --checkpoint_path $checkpoint_path --data_root_dir $data_root_dir --dataset $dataset --fixed_prompt_type industrial --epoch $epoch --model_version $model_version
# done

# Medical classification datasets - using medical prompts
# for dataset in headct brainmri br35h; do
#     python test.py --models_dir $models_dir --checkpoint_path $checkpoint_path --data_root_dir $data_root_dir --dataset $dataset --fixed_prompt_type medical --epoch $epoch --model_version $model_version
# done
