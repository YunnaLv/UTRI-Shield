from torch.utils.data import DataLoader
from utils.tools import *

transform = transforms.Compose([transforms.Resize(255),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(224),
                                transforms.ToTensor()])

train_dataset = ImageList('', open('data/CASIA/database.txt').readlines(), transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1)


def compute_mean_and_std(loader):
    batch_mean, batch_squared_mean, batch_num = 0, 0, 0
    for data in loader:
        data = data[0]
        batch_mean += torch.mean(data, dim=[0, 2, 3])
        batch_squared_mean += torch.mean(data ** 2, dim=[0, 2, 3])
        batch_num += 1
    mean = batch_mean / batch_num
    std = (batch_squared_mean / batch_num - mean ** 2) ** 0.5
    return mean, std


if __name__ == '__main__':
    mean, std = compute_mean_and_std(train_loader)
    print('mean =', mean)
    print('std =', std)
