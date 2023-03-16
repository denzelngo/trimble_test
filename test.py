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
    parser.add_argument('--data', default='dataset/test_images', type=str, help='initial pretrained weights path')
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

    # Create list of image paths
    img_files = sorted(os.listdir(arg.data), key=lambda x: int(float(x[:-4])))

    # Start testing and save the result in file result.csv
    with torch.no_grad(), open('result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['File name', 'Class'])
        for img_file in tqdm(img_files):
            img_pth = join(arg.data, img_file)
            img = cv2.imread(img_pth)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = transform(img).unsqueeze(0)
            if use_cuda:
                img = img.cuda()
            out = model(img)
            pred = torch.sigmoid(out)
            if pred > 0.5:
                cls = 'road'
            else:
                cls = 'field'
            writer.writerow([img_file, cls])
    print('Finished! Result saved in result.csv')
