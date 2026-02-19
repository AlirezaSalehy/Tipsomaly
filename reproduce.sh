# This script provides a sample command to test given checkpoints on a dataset
# For testing multiple datasets in a loop, refer to script train_test.sh

models_dir="/kaggle/working/tips"
data_root_dir="/kaggle/working/datasets"
model_version='l14h' # model version name, like s14h, l14h, g14h for TIPS model and google/siglip2-large-patch16-512 for SigLIP2 model 
checkpoint_path="./workspaces/trained_on_mvtec_default/vegan-arkansas/checkpoints"
# visa checkpoint at: './workspaces/trained_on_visa_default/vegan-arkansas/checkpoints'

### Test using industrial fixed prompts

# test on VisA
python test.py --models_dir $models_dir --checkpoint_path $checkpoint_path --data_root_dir $data_root_dir --dataset visa --epoch 2 --model_version $model_version --fixed_prompt_type industrial

# test on MVTec
# python test.py --models_dir $models_dir --checkpoint_path $checkpoint_path --data_root_dir $data_root_dir --dataset mvtec --epoch 2 --model_version $model_version --fixed_prompt_type industrial

# test on a medical dataset with industrial prompts
# python test.py --models_dir $models_dir --checkpoint_path $checkpoint_path  --data_root_dir $data_root_dir --dataset cvc-colondb --fixed_prompt_type industrial --epoch 2 --model_version $model_version

# test on a medical dataset with medical prompts
# python test.py --models_dir $models_dir --checkpoint_path $checkpoint_path  --data_root_dir $data_root_dir --dataset headct --fixed_prompt_type medical --epoch 2 --model_version $model_version