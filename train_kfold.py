import timm
import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='resnet18', type=str, help='model architecture, in timm.list_models()')
    parser.add_argument('--data', type=str, default='dataset/train', help='splitted train/val dataset path')
    parser.add_argument('--k', default=5, type=int, help='number of folds')
    arg = parser.parse_args()

    # Training settings
    MODEL_NAME = arg.arch
    IMAGE_SIZE = 256
    conf_thres = 0.5
    num_folds = arg.k
    batch_size = 16
    lr = 0.0001
    num_epochs = 5

    # Check cuda
    use_cuda = torch.cuda.is_available()

    # Resize and transform tensor to CHW format
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = datasets.ImageFolder(arg.data, transform=train_transform)

    # K-fold, set a fixed random_state for every model training
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)

    # Training setting: loss function
    criterion = nn.BCEWithLogitsLoss()

    # List to store accuracy and loss value
    folds_acc = []
    folds_f1 = []

    # Start k-fold cross-validation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print('Fold: ', fold)

        # Initialize/Reset model
        model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)
        if use_cuda:
            model.to('cuda')

        # Split dataset into K folds: 1 fold for validation, K-1 folds for training
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)
        trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        testloader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        # Initialize/Reset optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training
        print(f'Start K-fold cross-validation with {num_folds} folds.')
        for epoch in range(0, num_epochs):
            print('Epoch: ', epoch)
            model.train()
            train_running_loss = []
            train_running_correct = []
            n = 0
            for i, data in enumerate(trainloader):
                optimizer.zero_grad()
                img, target = data
                n += len(target)
                if use_cuda:
                    img = img.cuda()
                    target = target.cuda()
                target = target.view(-1, 1).float()

                outputs = model(img)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                # Compute accuracy
                pred = torch.sigmoid(outputs)
                num_correct = ((pred > conf_thres) == target).sum().item()
                train_running_correct.append(num_correct)
                train_running_loss.append(loss.item())

            print('train_loss: ', sum(train_running_loss) / len(train_running_loss))
            print('train_acc: ', sum(train_running_correct) / n)

        # Validation, using accuracy and F1-score
        model.eval()
        valid_running_loss = []
        valid_running_correct = []
        val_running_tp = 0
        val_running_fp = 0
        val_running_fn = 0
        n = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                img, target = data
                n += len(target)
                if use_cuda:
                    img = img.cuda()
                    target = target.cuda()

                target = target.view(-1, 1)
                outputs = model(img)
                loss = criterion(outputs, target.float())

                pred = torch.sigmoid(outputs)
                num_correct = ((pred > conf_thres) == target).sum().item()
                valid_running_correct.append(num_correct)
                valid_running_loss.append(loss.item())
                val_running_tp += (torch.logical_and((pred >= conf_thres) == 1, target == 1)).sum().item()
                val_running_fp += (torch.logical_and((pred >= conf_thres) == 1, target == 0)).sum().item()
                val_running_fn += (torch.logical_and((pred >= conf_thres) == 0, target == 1)).sum().item()
        precision = val_running_tp / (val_running_tp + val_running_fp + 1e-5)
        recall = val_running_tp / (val_running_tp + val_running_fn + 1e-5)
        f1_val = 2 * (precision * recall) / (precision + recall + 1e-5)

        val_loss = sum(valid_running_loss) / len(valid_running_loss)
        val_acc = sum(valid_running_correct) / n
        folds_acc.append(val_acc)
        folds_f1.append(f1_val)
        print('val_loss: ', val_loss)
        print('val_acc: ', val_acc)
        print('F1 val: ', f1_val)

    for i in range(num_folds):
        print(f'Fold {i}: acc: {folds_acc[i]},f1: {folds_f1[i]}')
    print('Average acc: ', sum(folds_acc) / len(folds_acc))
    print('Average F1: ', sum(folds_f1) / len(folds_f1))
