model_version='google/siglip2-large-patch16-512' # 'l14h' # l14h or l14
backbone_name='siglip2-hf' # 'l14h' # l14h or l14

model_name='siglip_test_temp' # $1
log_dir='' # $2

device='0' # $3
epoch='0' # $4

python test.py --dataset mvtec --devices $device --model_version $model_version --backbone_name $backbone_name --fixed_prompt_type industrial --image_size 512 --k_shot 0
# for dataset in visa mpdd btad sdd; do
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python test.py --dataset $dataset --devices $device --backbone_name $backbone_name --epoch $epoch --model_version $model_version --fixed_prompt_type object_agnostic --image_size 512
# done

# python test.py --dataset mvtec --devices $device --epoch $epoch --model_version $model_version --backbone_name $backbone_name --fixed_prompt_type industrial
# for dataset in visa mpdd btad sdd; do
#     OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python test.py --checkpoint_path ./workspaces/trained_on_mvtec_$model_name/$log_dir/checkpoints --dataset $dataset --devices $device  --epoch $epoch --model_version $model_version --fixed_prompt_type industrial
# done