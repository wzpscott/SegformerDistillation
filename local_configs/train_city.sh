python tools/convert_datasets/cityscapes.py data/cityscapes;
bash tools/dist_train.sh local_configs/10.16/City_s.py 8;
bash tools/dist_train.sh local_configs/10.16/City_cg10.py 8;
bash tools/dist_train.sh local_configs/10.16/City_cg3.py 8;
bash tools/dist_train.sh local_configs/10.16/City_c.py 8;
bash tools/dist_train.sh local_configs/10.16/City_cg5.py 8;