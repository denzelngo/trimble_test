import timm
import torch
import cv2
import os
import csv
import argparse
from torchvision import transforms
from tqdm import tqdm
from os.path import join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='dataset/test_images/1.jpeg', type=str, help='path to image')
    parser.add_argument('--weight', default='resnet18_best.pt', type=str, help='model trained weight path')
    arg = parser.parse_args()

    MODEL_NAME = 'resnet18'
    IMAGE_SIZE = 256

    # Check cuda
    use_cuda = torch.cuda.is_available()

    # Transform tensor to CHW format
    transform = transforms.Compose([transforms.ToTensor()])

    # Load model
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(arg.weight))
    model.eval()
    if use_cuda:
        model.cuda()

    # Read image
    if not os.path.isfile(arg.img):
        raise ValueError(f'Image not found at {arg.img}')
    img = cv2.imread(arg.img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = transform(img).unsqueeze(0)

    # Inference
    if use_cuda:
        img = img.cuda()
    out = model(img)
    pred = torch.sigmoid(out)
    if pred > 0.5:
        print('The class of image: road')
    else:
        print('The class of image: field')

