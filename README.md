# Trimble technical test: Field/Road classification
## Approach
I use pytorch and the models are taken from [timm](https://github.com/huggingface/pytorch-image-models). K-fold cross-validation is used to select the model.

## Installation
Install dependencies:
```
pip install -r requirements.txt
```
## K-fold cross-validation
Copy the training dataset into `dataset/train`. Run:
```
python train_kfold.py --arch resnet18 --k 5 --data dataset/train
```
- `--arch`: name of model architecture (**resnet18**, **efficientnet_b0**, **mobikenetv2_100**, ...). Use `timm.list_models()` to see the full list of architecture names.
- `--k`: number of folds
- `--data`: path to training dataset
## Data splitting
After cross-validation and model selection, we will split the dataset into training set and validation set to prepare for the model training. Run:
```
python split_dataset.py --data data dataset/train --val-size 0.2
```
- `--data`: path to training dataset
- `--val-size`: percentage of validation set

## Training
Run:
```
python train.py  --epochs 10 --batch-size 16 --data dataset  --conf 
```
- `--epoch`: number of training epochs
- `--batch-size`: batch size
- `--data`: path to dataset root
- `--conf`: classification confidence threshold

The learning curve and the trained models will be save at `runs/[date_time]` folder.
  

**Trained resnet18 model**: [download here](https://drive.google.com/file/d/1rp4ivfkwHkBr6SXC3saHRvllMIXzkAmy/view?usp=sharing)
## Test
After training the model, we can test the model on a separate test dataset.
```
python test.py --weight [trained_weight_file_path] --data dataset/test_images
```
- `--weight`: path to trained weight
- `--data`: path to the test dataset

The result will be saved at file `resul.csv`.

## Demo
You can run a demo to classify an image:
```
python inference.py --weight [trained_weight_file_path] --img [path_to_image]
```
- `--weight`: path to trained weight
- `--img`: path to the image
