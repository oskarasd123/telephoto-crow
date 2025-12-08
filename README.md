# telephoto-crow
this repository contains the model training and inference code for my crow detection project






# Getting started

Make sure you have all libraries in requirements.txt


download datasets:

`kaggle datasets download -d saisanjaykottakota/142-birds-species-object-detection-v1`

`kaggle datasets download -d wenewone/cub2002011`

unzip them:

`unzip 142-birds-species-object-detection-v1.zip -d data1/`

`unzip cub2002011.zip -d data2/`

try to load the dataset(optional): `python3 dataloader.py`

## Training

Pretrain a model: `python3 train.py`

Train the pretrained model on my data with: `python3 post_train.pt`

make graphs with `python3 graphs_and_statistics.py`
