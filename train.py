import timm
import torch
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from os.path import join
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=None, type=str, help='initial pretrained weights path')
    parser.add_argument('--data', type=str, default='dataset', help='splitted train/val dataset path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training/validation')
    parser.add_argument('--conf', type=float, default=0.5, help='classification confidence threshold')
    arg = parser.parse_args()

    # Training settings
    MODEL_NAME = 'resnet18'
    IMAGE_SIZE = 256
    epochs = arg.epochs
    batch_size = arg.batch_size
    conf = arg.conf

    # Check cuda
    use_cuda = torch.cuda.is_available()

    # Path to save output
    save_dir_root = 'runs/'
    if not os.path.isdir(save_dir_root):
        os.mkdir(save_dir_root)

    # Create a folder to save checkpoint, based on the date time of training
    folder_name = datetime.now().strftime("%d%B%Y_%H_%M")
    save_dir = join(save_dir_root, folder_name)
    os.mkdir(save_dir)

    # Create dataset and dataloader
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.RandomHorizontalFlip(p=0.2),
        # transforms.RandomRotation(20),
        # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ToTensor(),
    ])
    # No data augmentation for validation set
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader
    train_path = join(arg.data, 'train')
    val_path = join(arg.data, 'val')
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print(f'{len(train_loader.dataset)} images in training set')
    print(f'{len(val_loader.dataset)} images in validation set')

    # Initialize model
    if arg.pretrained:  # Load pretrained weight
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
        model.load_state_dict(torch.load(arg.pretrained))
        print(f'Loaded pretrained model from {arg.pretrained}')
    else:  # Initialize from weight trained on Imagenet
        model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)
        print('Model weight initialized from training on Imagenet.')

    # Training setting: loss function, optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    if use_cuda:
        model.cuda()

    # Metric to save the best model
    best = 0

    # Lists used to visualize the learning curve
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    # Start training
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        # Training
        model.train()
        train_running_loss = []
        train_running_correct = []
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            img, target = data
            if use_cuda:
                img = img.cuda()
                target = target.cuda()
            target = target.view(-1, 1).float()
            out = model(img)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            pred = torch.sigmoid(out)
            num_correct = ((pred > conf) == target).sum().item()
            train_running_correct.append(num_correct)
            train_running_loss.append(loss.item())

        train_loss = sum(train_running_loss) / len(train_running_loss)
        train_acc = sum(train_running_correct) / len(train_loader.dataset)
        print('train_loss: ', train_loss)
        print('train_acc: ', train_acc)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # Validation
        model.eval()
        valid_running_loss = []
        valid_running_correct = []
        val_running_tp = 0
        val_running_fp = 0
        val_running_fn = 0
        with torch.no_grad():
            for data in val_loader:
                img, target = data
                target = target.view(-1, 1).float()
                if use_cuda:
                    img = img.cuda()
                    target = target.cuda()
                outputs = model(img)
                loss = criterion(outputs, target)

                # Compute metrics: accuracy, F1 score
                pred = torch.sigmoid(outputs)
                num_correct = ((pred > conf) == target).sum().item()
                valid_running_correct.append(num_correct)
                valid_running_loss.append(loss.item())
                val_running_tp += (torch.logical_and((pred >= conf) == 1, target == 1)).sum().item()
                val_running_fp += (torch.logical_and((pred >= conf) == 1, target == 0)).sum().item()
                val_running_fn += (torch.logical_and((pred >= conf) == 0, target == 1)).sum().item()
        precision = val_running_tp / (val_running_tp + val_running_fp + 1e-5)
        recall = val_running_tp / (val_running_tp + val_running_fn + 1e-5)
        f1_val = 2 * (precision * recall) / (precision + recall + 1e-5)
        val_loss = sum(valid_running_loss) / len(valid_running_loss)
        val_acc = sum(valid_running_correct) / len(val_loader.dataset)
        print('val_loss: ', val_loss)
        print('val_acc: ', val_acc)
        print('f1_val: ', f1_val)
        val_acc_hist.append(val_acc)
        val_loss_hist.append(val_loss)

        # Save the best model
        if f1_val > best:
            best = f1_val
            torch.save(model.state_dict(), join(save_dir, MODEL_NAME + '_best.pt'))

        # Visualize the learning curve
        fig = plt.figure(figsize=(20, 10))
        plt.title("Learning curve")
        plt.plot(train_acc_hist, label='train_acc')
        plt.plot(val_acc_hist, label='val_acc')
        plt.plot(train_loss_hist, label='train_loss')
        plt.plot(val_loss_hist, label='val_loss')

        plt.xlabel('num_epochs', fontsize=12)
        plt.legend(loc='best')
        plt.savefig(join(save_dir, 'learning_curve.png'))

    # End training
    torch.save(model.state_dict(), join(save_dir, MODEL_NAME + '_last.pt'))
    print(f'Training finished. Trained model saved in {save_dir}.')
