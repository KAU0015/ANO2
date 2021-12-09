import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import os
import torchvision.models as models
import time


transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
])

batch_size = 32
data_dir = 'train_images'
image_datasets = datasets.ImageFolder(data_dir, transform=transform)
data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=12)
#print(image_datasets)

classes = ('free', 'full')

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def main():
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    #imshow(torchvision.utils.make_grid(images))
    net = models.resnet18(pretrained=True)

    for param in net.parameters():
        param.requires_grad = False

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 2)

    for name,param in net.named_parameters():
        if(("fc" in name) or ("layer4" in name)):
            param.requires_grad = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    params_to_update = [param for param in net.parameters() if param.requires_grad == True]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params_to_update, lr=0.0001, momentum=0.9)
    start = time.time()

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    print('Finished Training')
    end = time.time()
    train_time = end - start
    print(f"Training time (s): {round(train_time, 5)}")
    PATH = './resnet_pretrained_1_epochs_batch_32_lr0001.pth'
    torch.save(net, PATH)


if __name__ == '__main__':
    main()
#((n+2*p-f)/s)+1