for dataset in visa mpdd btad sdd dagm dtd; do
    python test.py --dataset $dataset --devices 3
done