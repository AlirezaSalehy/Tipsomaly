device=2

# for dataset in mvtec visa mpdd btad sdd dagm dtd; do
#     python test.py --dataset $dataset --devices $device
# done

# for dataset in brainmri headct br35h isic tn3k cvc-colondb cvc-clinicdb; do
#     python test.py --dataset_category med --dataset $dataset --devices $device 
# done

for dataset in brainmri headct br35h; do
    python test.py --dataset_category med --dataset $dataset --devices $device 
done