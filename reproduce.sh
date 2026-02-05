# This script provides a sample command to test given checkpoints on a dataset
# For testing multiple datasets in a loop, refer to script train_test.sh

models_dir="./"
data_root_dir="/kaggle/input/"
model_version='l14h' # model version name, like s14h, l14h, g14h for TIPS model and google/siglip2-large-patch16-512 for SigLIP2 model 
checkpoint_path="/kaggle/working/checkpoints/"

### Test using industrial fixed prompts

# test TIPS on VisA (trained on MVTec)
python test.py --models_dir $models_dir --checkpoint_path $checkpoint_path --data_root_dir $data_root_dir --dataset visa --devices 0  --epoch 2 --model_version $model_version --fixed_prompt_type industrial

# test TIPS on MVTec (trained on VisA)
# python test.py --models_dir $models_dir --checkpoint_path $checkpoint_path --data_root_dir $data_root_dir --dataset mvtec --devices 0 --epoch 2 --model_version $model_version --fixed_prompt_type industrial

# test TIPS on a medical dataset with industrial prompts
# python test.py --models_dir $models_dir --checkpoint_path $checkpoint_path  --data_root_dir $data_root_dir --dataset cvc-colondb --dataset_category med --devices 0 --fixed_prompt_type industrial --epoch 2 --model_version $model_version

# test TIPS on a medical dataset with medical prompts
# python test.py --models_dir $models_dir --checkpoint_path $checkpoint_path  --data_root_dir $data_root_dir --dataset headct --dataset_category med --devices 0 --fixed_prompt_type medical --epoch 2 --model_version $model_version

