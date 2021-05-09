python train.py -data_pkl ./vi_data.pkl \
-embs_share_weight -proj_share_weight -label_smoothing \
-output_dir checkpoint -b 32 -warmup=8000 -epoch 100