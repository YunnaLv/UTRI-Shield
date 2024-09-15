import torch
from torch import nn
from network import Vgg, ResNet
from utils.tools import ImageList, CalcTopMap, image_transform
from utils.noise_utils import *
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms
from DiffJPEG import DiffJPEG
import os
import torchvision.models as models

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['TORCH_HOME'] = '/data/UTAH_code/UTAP_robust/model/torch-model'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

cpu_num = 1     # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

# 模型加载
def load_model(hash_bit, specific_model, model_path):
    if 'ResNet' in specific_model:
        model = ResNet(hash_bit, specific_model)  # hash_bit and specific model type
    elif 'Vgg' in specific_model:
        model = Vgg(hash_bit, specific_model)
    # else:
    #     model = AlexNet(hash_bit)
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.eval()
    return model


# 数据加载(预处理 + loader)
def load_data(data_path, list_path, batch_size, resize_size, crop_size, data, dset):
    dataset = ImageList(data_path, open(list_path).readlines(),
                        transform=image_transform(resize_size, crop_size, data, dset))
    ### 原来是 shuffle = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return dataloader

# resize
def image_resize(images, new_size):
    # new_size = 100  # ?
    transform_resize = transforms.Compose([
        transforms.Resize((new_size, new_size)),
        transforms.Resize((224, 224)),
    ])
    return transform_resize(images)

# 模糊
def image_GaussianBlur(images, ker, cir_len):
    # ker, cir_len = 3, 30     # ?  3 50/ 5 50
    # ker, cir_len = 5, 50
    transform_blur = transforms.Compose([
        transforms.GaussianBlur(ker, cir_len)
        # transforms.GaussianBlur(ker)
    ])
    return transform_blur(images)

# 高斯噪声
def image_noise(images, noise_v):
    # noise_v = 0.002 ** 0.5  # ?
    # noise_v = 0.005 ** 0.5
    mask = np.random.normal(0, noise_v, (3, 224, 224))
    mask = torch.from_numpy(mask).float()
    return images + mask

# 压缩
def image_jpeg(images, quality, dataset):
    # quality = 80    # ? 越大压缩程度越小
    # quality = 50
    # quality = 10
    jpeg = DiffJPEG(height=224, width=224, differentiable=True, quality=quality)
    return jpeg(images, dataset)

# img + noise 计算net(dataloader + noise)结果
def compute_result(save_path, llist, dataloader, noise, net, device, dataset):
    bs, bs_2, clses = [], [], []
    net.eval()
    idx = 0
    for img, cls, _ in tqdm(dataloader):
        perturbated_images = clamp_img(img + noise, dataset)

        # 存到save_path
        save_path_ = save_path + llist[idx]
        save_path_ = save_path_.replace('.jpg', '.png')
        print(save_path_)
        idx += 1
        # save_image(un_normalize(img[0], dataset), save_path_)
        save_image(un_normalize(perturbated_images[0], dataset), save_path_)
        # return

        bs.append((net(perturbated_images.to(device))).data.cpu())  # 扰动图像
        bs_2.append((net(img.to(device))).data.cpu())  # 原图
        clses.append(cls)
    return torch.cat(bs).sign(), torch.cat(bs_2).sign(), torch.cat(clses)

if __name__ == '__main__':
    ######## 64bit CSQ
    save_path1 = "/data/UTAH_save/CSQ/ResNet34/CASIA/0.8795945076653223/"  # CSQ-ResNet34
    save_path2 = "/data/UTAH_save/CSQ/ResNet50/CASIA/0.8828318460072873/"  # CSQ-ResNet50
    save_path3 = "/data/UTAH_save/CSQ/Vgg16/CASIA/0.8504923177106464/"  # CSQ-Vgg16
    save_path4 = "/data/UTAH_save/CSQ/Vgg19/CASIA/0.8237669211806679/"  # CSQ-Vgg19

    model1 = load_model(64, 'ResNet34', save_path1 + "model.pt")
    model1 = model1.cuda()

    # 数据集
    dset = "CASIA"
    # 加载测试集
    data_path = ''
    list_path = '/data/UTAH_code/UTAP/UTAP/user_data/10242048@qq.com/2024-08-03-08-12-38/train_img/train.txt'
    test_loader = load_data(data_path, list_path, 1, 256, 224, 'test', dset)
    llist = []

    # 打开文件并按行读取内容
    with open(list_path, 'r') as f:
        for line in f:
            # 去除每行末尾的换行符
            line = line.strip().split(' ')[0]
            line1 = line.split('/')[-2]
            line2 = line.split('/')[-1]
            l = line1 + '/' + line2

            print(l)
            # 将读取的内容添加到列表中
            llist.append(l)

    # 打印读取的内容列表
    print(llist)

    # 加载noise
    noise_path = '/data/UTAH_code/UTAP/UTAP/exp/CSQ/Vgg16/CASIA/UTAP_0/'
    save_path = '/data/UTAH_code/UTAP/UTAP/save_img/org_2/'
    noise_ = np.load(noise_path + 'best_noise.npy')
    noise = torch.from_numpy(noise_)
    noise = clamp_noise(noise, dset)

    topk = 300
    # for m in range(17):
    # for m in [1]:
    #     print('============ mode =', m, '============')

    compute_result(save_path, llist, test_loader, noise, model1, device="cuda:0", dataset=dset)