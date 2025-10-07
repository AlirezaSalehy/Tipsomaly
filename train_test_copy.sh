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

model_version='l14h' # 'l14h' # l14h or l14

model_name=$1
log_dir=$2

device=$3
epoch=$4

# for dataset in headct brainmri br35h; do
#     OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints  --dataset $dataset --devices $device --fixed_prompt_type medical_low_1 --epoch $epoch --model_version $model_version
# done

# for dataset in headct brainmri br35h; do
#     OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints  --dataset $dataset --devices $device --fixed_prompt_type medical_low_2 --epoch $epoch --model_version $model_version
# done

# for dataset in headct brainmri br35h; do
#     OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints  --dataset $dataset --devices $device --fixed_prompt_type medical_low_3 --epoch $epoch --model_version $model_version
# done

for dataset in headct brainmri br35h; do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints  --dataset $dataset --devices $device --fixed_prompt_type medical_low_4 --epoch $epoch --model_version $model_version
done
