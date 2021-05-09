# A Pytorch Transformer Simple Implementation

# Installation
    pip install -r requirements.txt
# Usage

### 1) Preprocess
```
python preprocess.py -share_vocab -save_data vi_data.pkl
```

### 2) Train the model
```
python train.py -data_pkl ./vi_data.pkl \
-embs_share_weight -proj_share_weight -label_smoothing \
-output_dir checkpoint -b 32 -warmup=8000 -epoch 100
```

### 3) Test the model
```bash
python translate.py -data_pkl vi_data.pkl -model ./output/model.chkpt -output prediction.txt
```

# Result



| Model                   | R1            | R2            | RL        |
| :---                    |     :---:     |     :---:     |     :---: |
| SEGMENT                 | 35.95         | 10.95         | 28.33     |
| KeyBERT(phobert-base)   | 45.46         | 22.79         | 35.21     |
| Transformer small       | **51.61**     | **30.79**     | **44.11** |  




# Acknowledge 
Most of the code is referenced in [`here`](https://github.com/jadore801120/attention-is-all-you-need-pytorch)